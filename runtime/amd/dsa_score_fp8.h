/* dsa_score_fp8.h — op 117 (DSA prefill indexer score) with vLLM 0.29 ROCm's FP8 arithmetic
 * (PLOW_DSA_IDX_FP8), quantized on the fly from the packet's bf16 operands (no packet change):
 *   k : per position,   ks = ue8m0(max(amax, 1e-4) / 448), k8 = e4m3fn(k / ks)
 *   q : per (row, head), qs = ue8m0(max(amax / 448, 1e-10)), q8 = e4m3fn(q / qs)
 *   w'[h] = ((w[h] * qs[h]) * 128^-0.5) * 32^-0.5            (f32, in that order)
 *   score[t][j] = ks[j] * sum_h w'[h] * relu(q8[h] . k8[j])  (x64 e4m3 MFMA, f32 accumulate)
 * ue8m0(x) = 2^ceil(log2(x)) of the correctly rounded quotient. The head sum replays the exact
 * operation order of the pinned image's aiter gluon kernel (head_sum2 below). The packet scale
 * (fj[0]) is not read: w' rounds the two factors separately.
 *
 * Decomposition: a pack is NW*2 query rows (wave w owns rows p*PACK + w and p*PACK + NW + w); a
 * work item is (pack, key span), longest packs first, dealt to workgroups in snake order. Key
 * slabs of TILE_N positions are quantized to e4m3 as they are staged, double-buffered in LDS, one
 * barrier per slab; the next slab is staged under the slab's first MFMAs and each 32-key
 * subtile's epilogue runs under the next subtile's MFMAs. */
#pragma once

namespace dsa_fp8 {
constexpr unsigned DI = 128, HI = 32, KS8 = DI + 16;

template <unsigned TILE_N>
constexpr unsigned lds_bytes() { return 2u * (TILE_N * KS8 + TILE_N * 4u); }

/* 2^ceil(log2(a / 448)) for a normal `a` that is a bf16 value or 1e-4f: with a = m * 2^e and
 * 448 = 1.75 * 2^8 it is 2^(e-8) if m <= 1.75, else 2^(e-7) — the mantissa carry of
 * a + (0x7fffff - 0x600000) — and a bf16/1e-4f `a` is never within an f32 ulp above 1.75 * 2^e,
 * so this equals the correctly rounded quotient's ceiling. */
__device__ __forceinline__ float ue8m0_div448(float a) {
    return __uint_as_float(((__float_as_uint(a) + 0x1fffffu) & 0xff800000u) - (8u << 23));
}
/* max over each row of 16 lanes (non-negative values, so as u32) */
__device__ __forceinline__ unsigned row16_max(unsigned v) {
    v = max(v, (unsigned)__builtin_amdgcn_update_dpp(0, (int)v, 0xb1, 0xf, 0xf, true));
    v = max(v, (unsigned)__builtin_amdgcn_update_dpp(0, (int)v, 0x4e, 0xf, 0xf, true));
    v = max(v, (unsigned)__builtin_amdgcn_update_dpp(0, (int)v, 0x141, 0xf, 0xf, true));
    v = max(v, (unsigned)__builtin_amdgcn_update_dpp(0, (int)v, 0x140, 0xf, 0xf, true));
    return v;
}
/* |bf16| max of two packed pairs as packed u16 (bf16 magnitudes order as integers) */
typedef unsigned short u16x2 __attribute__((ext_vector_type(2)));
__device__ __forceinline__ unsigned amax2(unsigned m, unsigned x) {
    return __builtin_bit_cast(unsigned, __builtin_elementwise_max(__builtin_bit_cast(u16x2, m),
                                                                  __builtin_bit_cast(u16x2, x & 0x7fff7fffu)));
}
/* packed |bf16| max -> f32 */
__device__ __forceinline__ float amax_f(unsigned m) { return __uint_as_float(max(m & 0xffffu, m >> 16) << 16); }
/* four bf16 (two packed pairs) -> four e4m3 of x / s; bit-identical to cvt_pk_fp8_f32(x / s) for a
 * power-of-two s (probed over 1M pairs) */
typedef short i16x2 __attribute__((ext_vector_type(2)));
typedef __bf16 bf16x2v __attribute__((ext_vector_type(2)));
__device__ __forceinline__ unsigned pk4(unsigned x, unsigned y, float s) {
    i16x2 r = __builtin_amdgcn_cvt_scalef32_pk_fp8_bf16(i16x2{0, 0}, __builtin_bit_cast(bf16x2v, x), s, false);
    r = __builtin_amdgcn_cvt_scalef32_pk_fp8_bf16(r, __builtin_bit_cast(bf16x2v, y), s, true);
    return __builtin_bit_cast(unsigned, r);
}
/* one v_max_i32: fmaxf(x, 0) lowers to a canonicalize + max pair. Negative floats (and -0) are
 * negative ints; MFMA sums are never NaN. */
__device__ __forceinline__ float relu(float x) { return __int_as_float(max(__float_as_int(x), 0)); }
typedef float f32x2 __attribute__((ext_vector_type(2)));
/* The pinned aiter gluon kernel's per-lane head sum (its gfx950 ISA, NUM_CHAINS=0 since that
 * Triton lacks the folded reduction): s = w1*r1; s = fma(w0, r0, s); s = fma(wi, ri, s) for
 * slots 2..9; then s += round(wi*ri) for slots 10..15 (those products were packed muls, so not
 * contracted). Rows 0 and 1 of the wave ride the two halves of each packed op. */
__device__ __forceinline__ f32x2 head_sum2(const f32x16& a0, const f32x16& a1, const f32x2* w) {
#pragma clang fp contract(off)
    auto r = [&](int i) { return f32x2{relu(a0[i]), relu(a1[i])}; };
    f32x2 s = w[1] * r(1);
    s = __builtin_elementwise_fma(w[0], r(0), s);
#pragma unroll
    for (int i = 2; i < 10; i++) s = __builtin_elementwise_fma(w[i], r(i), s);
#pragma unroll
    for (int i = 10; i < 16; i++) s = s + w[i] * r(i);
    return s;
}
}  // namespace dsa_fp8

template <unsigned TILE_N, int QPW>
__device__ void d_index_score_pf_fp8(float* __restrict__ Score, const bf16* __restrict__ Qidx,
                                     const bf16* __restrict__ Kidx, const bf16* __restrict__ W,
                                     const int* __restrict__ kv_len, unsigned n_tok,
                                     unsigned kv_stride, unsigned slice, unsigned nblk,
                                     unsigned char* lds) {
    using namespace dsa_fp8;
    static_assert(TILE_N % 32u == 0, "slab is walked in 32-key MFMA subtiles");
    static_assert(QPW == 2, "rows are reduced in lane^32 pairs");
    constexpr unsigned NW = PLOW_THREADS / 64u, PACK = NW * QPW;
    constexpr unsigned SLAB = TILE_N * KS8 + TILE_N * 4u;
    constexpr unsigned NPRE = TILE_N * (DI / 8u) / PLOW_THREADS;
    constexpr unsigned NST = TILE_N / 32u;
    static_assert(NPRE * PLOW_THREADS == TILE_N * (DI / 8u), "slab splits evenly");
    auto* const Sc = as_glob(Score);
    const unsigned tid = threadIdx.x, lane = tid & 63u, wave = tid >> 6;
    const unsigned frow = lane & 31u, kh = lane >> 5;
    const unsigned len = (unsigned)as_glob(kv_len)[0];
    if (n_tok == 0u) return;
    const unsigned q_pos0 = len - n_tok;
    const unsigned n_pack = (n_tok + PACK - 1u) / PACK;
    /* span: ~4 work items per workgroup over the causal area, never under 1024 keys */
    unsigned span = (unsigned)(((unsigned long long)n_pack * (len - n_tok / 2u)) / (4ull * nblk));
    span = span < 1024u ? 1024u : span;
    span = (span + TILE_N - 1u) / TILE_N * TILE_N;
    const unsigned n_span = (len + span - 1u) / span;
    const unsigned n_work = n_pack * n_span;

    /* buffer loads: out-of-range key positions / query rows read as zero, no branches */
    const __amdgpu_buffer_rsrc_t qrs = buf_rsrc_u(Qidx, n_tok * HI * DI);
    const __amdgpu_buffer_rsrc_t wrs = buf_rsrc_u(W, n_tok * HI);
    uint4 pre[NPRE];
    __amdgpu_buffer_rsrc_t krs;
    auto fetch = [&](unsigned b) {
#pragma unroll
        for (unsigned i = 0; i < NPRE; i++)
            pre[i] = __builtin_bit_cast(uint4, __builtin_amdgcn_raw_buffer_load_b128(
                                                   krs, (b * DI + (tid + i * PLOW_THREADS) * 8u) * 2u, 0, 0));
    };
    /* 16 consecutive lanes hold one key row (8 dims each) */
    auto stage = [&](unsigned char* buf) {
#pragma unroll
        for (unsigned i = 0; i < NPRE; i++) {
            const unsigned c = tid + i * PLOW_THREADS;
            const unsigned row = c / (DI / 8u), c8 = (c % (DI / 8u)) * 8u;
            const uint4 u = pre[i];
            const unsigned m = amax2(amax2(amax2(u.x & 0x7fff7fffu, u.y), u.z), u.w);
            const unsigned am = row16_max(max(m & 0xffffu, m >> 16));
            const float ks = ue8m0_div448(fmaxf(__uint_as_float(am << 16), 1e-4f));
            *(uint2*)(buf + row * KS8 + c8) = make_uint2(pk4(u.x, u.y, ks), pk4(u.z, u.w, ks));
            if (c8 == 0u) ((float*)(buf + TILE_N * KS8))[row] = ks;
        }
    };

    for (unsigned r = 0;; r++) {
        const unsigned k = r * nblk + ((r & 1u) ? nblk - 1u - slice : slice);
        if (k >= n_work) break;
        const unsigned p = n_pack - 1u - k / n_span, sp = k % n_span;
        unsigned pack_last = p * PACK + (PACK - 1u);
        if (pack_last >= n_tok) pack_last = n_tok - 1u;
        const unsigned pack_end = q_pos0 + pack_last + 1u;
        const unsigned s_lo = sp * span;
        if (s_lo >= pack_end) continue;
        const unsigned s_hi = (s_lo + span < pack_end) ? (s_lo + span) : pack_end;
        krs = buf_rsrc_u(Kidx, s_hi * DI);
        fetch(s_lo);

        /* A operands: aiter's k_width-16 layout, lane half kh holding dims 64*ks + 16*kh + [0, 16)
         * then 64*ks + 32 + 16*kh + [0, 16); the B read below uses the same slots */
        fp8v32 qa[2][2];
        f32x2 wv[16]; /* {row 0, row 1} weight of accumulator slot i */
        unsigned row_end[2];
#pragma unroll
        for (int j = 0; j < 2; j++) {
            const unsigned t = p * PACK + (unsigned)j * NW + wave;
            row_end[j] = t < n_tok ? (q_pos0 + t + 1u) : 0u;
            uint4 x[8];
#pragma unroll
            for (int c = 0; c < 8; c++)
                x[c] = __builtin_bit_cast(uint4, __builtin_amdgcn_raw_buffer_load_b128(
                    qrs, ((t * HI + frow) * DI + 64u * (c / 4) + 32u * ((c % 4) / 2) + 16u * kh + 8u * (c % 2)) * 2u,
                    0, 0));
            unsigned m = 0u;
#pragma unroll
            for (int c = 0; c < 8; c++) m = amax2(amax2(amax2(amax2(m, x[c].x), x[c].y), x[c].z), x[c].w);
            float amax = amax_f(m);
            amax = fmaxf(amax, __shfl_xor(amax, 32, PLOW_WAVE));
            const float qs = fmaxf(ue8m0_div448(fmaxf(amax, 0x1p-60f)), 0x1p-33f);
#pragma unroll
            for (int ks = 0; ks < 2; ks++)
#pragma unroll
                for (int c = 0; c < 4; c++) {
                    const uint4 v = x[ks * 4 + c];
                    qa[j][ks][2 * c] = (int)pk4(v.x, v.y, qs);
                    qa[j][ks][2 * c + 1] = (int)pk4(v.z, v.w, qs);
                }
            float wh = bf2f(__builtin_amdgcn_raw_buffer_load_b16(wrs, (t * HI + frow) * 2u, 0, 0)) * qs;
            wh *= 0.08838834764831845f;
            wh *= 0.1767766952966369f;
#pragma unroll
            for (int i = 0; i < 16; i++) wv[i][j] = __shfl(wh, (int)mfma_acc_m(lane, (unsigned)i), PLOW_WAVE);
        }
        const unsigned wave_end = __builtin_amdgcn_readfirstlane(row_end[0] > row_end[1] ? row_end[0] : row_end[1]);
        const unsigned my_end = kh ? row_end[1] : row_end[0];
        float* const my_row = &Sc[(size_t)(p * PACK + kh * NW + wave) * kv_stride];

        stage(lds);
        if (s_lo + TILE_N < s_hi) fetch(s_lo + TILE_N);
        __syncthreads();
        unsigned cur = 0u;
        for (unsigned b = s_lo; b < s_hi; b += TILE_N, cur ^= 1u) {
            const unsigned char* const kb = lds + cur * SLAB;
            const float* const ksc = (const float*)(kb + TILE_N * KS8);
            unsigned n_st = wave_end > b ? (wave_end - b + 31u) / 32u : 0u;
            n_st = n_st < NST ? n_st : NST;
            fp8v32 kf[2][2];
            auto ldk = [&](unsigned st, fp8v32* kq) {
                const unsigned char* const kr = kb + (32u * st + frow) * KS8 + 16u * kh;
#pragma unroll
                for (int ks = 0; ks < 2; ks++) {
                    const uint4 lo = *(const uint4*)(kr + 64 * ks), hi = *(const uint4*)(kr + 64 * ks + 32);
                    kq[ks] = fp8v32{(int)lo.x, (int)lo.y, (int)lo.z, (int)lo.w,
                                    (int)hi.x, (int)hi.y, (int)hi.z, (int)hi.w};
                }
            };
            auto mm = [&](const fp8v32* q, const fp8v32* kq) -> f32x16 {
                return plow_mfma_fp8_32x32(q[1], kq[1], plow_mfma_fp8_32x32(q[0], kq[0], (f32x16)(0.0f)));
            };
            /* row 0's sum lands in lanes 0-31 and row 1's in lanes 32-63 after the half swap */
            auto epi = [&](unsigned st, const f32x16* a) {
                const f32x2 s = head_sum2(a[0], a[1], wv);
                const auto sw = __builtin_amdgcn_permlane32_swap(__float_as_uint(s.x), __float_as_uint(s.y), false, false);
                const float tot = __uint_as_float(sw[0]) + __uint_as_float(sw[1]);
                const unsigned pos = b + 32u * st + frow;
                if (pos < my_end) st_act<float>(&my_row[pos], tot * ksc[32u * st + frow]);
            };
            f32x16 acc[2][2];
            if (n_st > 0u) {
                ldk(0u, kf[0]);
                acc[0][0] = mm(qa[0], kf[0]);
                acc[0][1] = mm(qa[1], kf[0]);
                if (n_st > 1u) ldk(1u, kf[1]);
            }
            if (b + TILE_N < s_hi) {
                stage(lds + (cur ^ 1u) * SLAB);
                if (b + 2u * TILE_N < s_hi) fetch(b + 2u * TILE_N);
            }
#pragma unroll
            for (unsigned st = 1; st < NST; st++) {
                if (st < n_st) {
                    acc[st & 1u][0] = mm(qa[0], kf[st & 1u]);
                    acc[st & 1u][1] = mm(qa[1], kf[st & 1u]);
                    if (st + 1u < n_st) ldk(st + 1u, kf[(st + 1u) & 1u]);
                    epi(st - 1u, acc[(st - 1u) & 1u]);
                }
            }
            if (n_st > 0u) {
                if ((n_st - 1u) & 1u) epi(n_st - 1u, acc[1]);
                else epi(n_st - 1u, acc[0]);
            }
            __syncthreads();
        }
    }
}
