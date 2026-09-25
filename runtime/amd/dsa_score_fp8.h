/* dsa_score_fp8.h — op 117 (DSA prefill indexer score) with vLLM 0.29 ROCm's FP8 arithmetic
 * (PLOW_DSA_IDX_FP8), quantized on the fly from the packet's bf16 operands (no packet change):
 *   k : per position,   ks = ue8m0(max(amax, 1e-4) / 448), k8 = e4m3fn(k / ks)
 *   q : per (row, head), qs = ue8m0(max(amax / 448, 1e-10)), q8 = e4m3fn(q / qs)
 *   w'[h] = ((w[h] * qs[h]) * 128^-0.5) * 32^-0.5            (f32, in that order)
 *   score[t][j] = ks[j] * sum_h w'[h] * relu(q8[h] . k8[j])  (x64 e4m3 MFMA, f32 accumulate)
 * ue8m0(x) = 2^ceil(log2(x)) of the correctly rounded quotient. The head sum follows aiter's gfx950
 * kernel (head_sum below), then the lane^32 half. The packet scale (fj[0]) is not read: w' rounds
 * the two factors separately.
 *
 * Decomposition: a pack is NW*QPW query rows (wave w owns rows p*PACK + j*NW + w); a work item
 * is (pack, key span), longest packs first, dealt to workgroups in snake order. Key slabs of
 * TILE_N positions are quantized to e4m3 as they are staged, double-buffered in LDS, one
 * barrier per slab; the next slab is staged under the slab's first MFMAs and each 32-key
 * subtile's epilogue runs under the next subtile's MFMAs. */
#pragma once

namespace dsa_fp8 {
constexpr unsigned DI = 128, HI = 32, KS8 = DI + 16;

template <unsigned TILE_N>
constexpr unsigned lds_bytes() { return 2u * (TILE_N * KS8 + TILE_N * 4u); }

__device__ __forceinline__ float pow2_ceil(float x) {
    return __builtin_bit_cast(float, (__builtin_bit_cast(unsigned, x) + 0x7fffffu) & 0xff800000u);
}
__device__ __forceinline__ float pow2_inv(float s) {
    return __builtin_bit_cast(float, 0x7f000000u - __builtin_bit_cast(unsigned, s));
}
/* 2^ceil(log2(fl(a / 448))). The reciprocal product can sit one ulp on the wrong side of a power
 * of two (a = 1.75 * 2^m, a bf16 value); 448 * s is exact, and for a bf16 or 1e-4f `a` the
 * quotient is never within an ulp above a power of two unless exact, so compare against it. */
__device__ __forceinline__ float ue8m0_div448(float a) {
    float s = pow2_ceil(a * (1.0f / 448.0f));
    if (224.0f * s >= a) s *= 0.5f;
    if (448.0f * s < a) s *= 2.0f;
    return s;
}
/* max over each row of 16 lanes */
__device__ __forceinline__ float row16_max(float v) {
    const int ninf = __float_as_int(-INFINITY);
    v = fmaxf(v, __int_as_float(__builtin_amdgcn_update_dpp(ninf, __float_as_int(v), 0xb1, 0xf, 0xf, true)));
    v = fmaxf(v, __int_as_float(__builtin_amdgcn_update_dpp(ninf, __float_as_int(v), 0x4e, 0xf, 0xf, true)));
    v = fmaxf(v, __int_as_float(__builtin_amdgcn_update_dpp(ninf, __float_as_int(v), 0x141, 0xf, 0xf, true)));
    v = fmaxf(v, __int_as_float(__builtin_amdgcn_update_dpp(ninf, __float_as_int(v), 0x140, 0xf, 0xf, true)));
    return v;
}
__device__ __forceinline__ unsigned pk4(float a, float b, float c, float d) {
    const unsigned lo = __builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0u, false);
    return __builtin_amdgcn_cvt_pk_fp8_f32(c, d, lo, true);
}
/* vLLM (aiter gluon, NUM_CHAINS=4) head reduction for one lane: chain c runs heads
 * 8c + 4*kh + {0..3} (accumulator slots 4c..4c+3) as a product then three FMAs; the chains are
 * summed in order. The lane^32 half is added by the caller. */
__device__ __forceinline__ float head_sum(const f32x16& a, const float* w) {
    float ch[4];
#pragma unroll
    for (int c = 0; c < 4; c++) {
        ch[c] = w[4 * c] * fmaxf(a[4 * c], 0.0f);
#pragma unroll
        for (int k = 1; k < 4; k++) ch[c] = __builtin_fmaf(w[4 * c + k], fmaxf(a[4 * c + k], 0.0f), ch[c]);
    }
    return ((ch[0] + ch[1]) + ch[2]) + ch[3];
}
__device__ __forceinline__ float bf_lo(unsigned u) { return __builtin_bit_cast(float, u << 16); }
__device__ __forceinline__ float bf_hi(unsigned u) { return __builtin_bit_cast(float, u & 0xffff0000u); }
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
            pre[i] = __builtin_bit_cast(
                uint4, __builtin_amdgcn_raw_buffer_load_b128(krs, (b * DI + tid * 8u + i * PLOW_THREADS * 8u) * 2u, 0, 0));
    };
    /* 16 consecutive lanes hold one key row (8 dims each) */
    auto stage = [&](unsigned char* buf) {
#pragma unroll
        for (unsigned i = 0; i < NPRE; i++) {
            const unsigned c = tid + i * PLOW_THREADS;
            const unsigned row = c / (DI / 8u), c8 = (c % (DI / 8u)) * 8u;
            const uint4 u = pre[i];
            const float f[8] = {bf_lo(u.x), bf_hi(u.x), bf_lo(u.y), bf_hi(u.y),
                                bf_lo(u.z), bf_hi(u.z), bf_lo(u.w), bf_hi(u.w)};
            float amax = 0.0f;
#pragma unroll
            for (int e = 0; e < 8; e++) amax = fmaxf(amax, fabsf(f[e]));
            amax = row16_max(amax);
            const float ks = ue8m0_div448(fmaxf(amax, 1e-4f)), inv = pow2_inv(ks);
            uint2 o;
            o.x = pk4(f[0] * inv, f[1] * inv, f[2] * inv, f[3] * inv);
            o.y = pk4(f[4] * inv, f[5] * inv, f[6] * inv, f[7] * inv);
            *(uint2*)(buf + row * KS8 + c8) = o;
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

        fp8v32 qa[2][2];
        float wv[2][16];
        unsigned row_end[2];
#pragma unroll
        for (int j = 0; j < 2; j++) {
            const unsigned t = p * PACK + (unsigned)j * NW + wave;
            const bool live = t < n_tok;
            row_end[j] = live ? (q_pos0 + t + 1u) : 0u;
            /* aiter's k_width-16 operand layout: lane half kh holds dims 64*ks + 16*kh + [0, 16)
             * then 64*ks + 32 + 16*kh + [0, 16); the B read below uses the same slots */
            uint4 x[8];
#pragma unroll
            for (int c = 0; c < 8; c++)
                x[c] = __builtin_bit_cast(uint4, __builtin_amdgcn_raw_buffer_load_b128(
                    qrs, ((t * HI + frow) * DI + 64u * (c / 4) + 32u * ((c % 4) / 2) + 16u * kh + 8u * (c % 2)) * 2u,
                    0, 0));
            float amax = 0.0f;
#pragma unroll
            for (int c = 0; c < 8; c++) {
                amax = fmaxf(amax, fmaxf(fabsf(bf_lo(x[c].x)), fabsf(bf_hi(x[c].x))));
                amax = fmaxf(amax, fmaxf(fabsf(bf_lo(x[c].y)), fabsf(bf_hi(x[c].y))));
                amax = fmaxf(amax, fmaxf(fabsf(bf_lo(x[c].z)), fabsf(bf_hi(x[c].z))));
                amax = fmaxf(amax, fmaxf(fabsf(bf_lo(x[c].w)), fabsf(bf_hi(x[c].w))));
            }
            amax = fmaxf(amax, __shfl_xor(amax, 32, PLOW_WAVE));
            const float qs = fmaxf(ue8m0_div448(amax), 0x1p-33f), inv = pow2_inv(qs);
#pragma unroll
            for (int ks = 0; ks < 2; ks++)
#pragma unroll
                for (int c = 0; c < 4; c++) {
                    const uint4 v = x[ks * 4 + c];
                    qa[j][ks][2 * c] = (int)pk4(bf_lo(v.x) * inv, bf_hi(v.x) * inv,
                                                bf_lo(v.y) * inv, bf_hi(v.y) * inv);
                    qa[j][ks][2 * c + 1] = (int)pk4(bf_lo(v.z) * inv, bf_hi(v.z) * inv,
                                                    bf_lo(v.w) * inv, bf_hi(v.w) * inv);
                }
            float wh = bf2f(__builtin_amdgcn_raw_buffer_load_b16(wrs, (t * HI + frow) * 2u, 0, 0)) * qs;
            wh *= 0.08838834764831845f;
            wh *= 0.1767766952966369f;
#pragma unroll
            for (int i = 0; i < 16; i++)
                wv[j][i] = __shfl(wh, (int)mfma_acc_m(lane, (unsigned)i), PLOW_WAVE);
        }
        const unsigned wave_end = row_end[0] > row_end[1] ? row_end[0] : row_end[1];
        const unsigned my_end = kh ? row_end[1] : row_end[0];
        float* const my_row = &Sc[(size_t)(p * PACK + kh * NW + wave) * kv_stride];

        stage(lds);
        if (s_lo + TILE_N < s_hi) fetch(s_lo + TILE_N);
        __syncthreads();
        unsigned cur = 0u;
        for (unsigned b = s_lo; b < s_hi; b += TILE_N, cur ^= 1u) {
            const unsigned char* const kb = lds + cur * SLAB;
            const float* const ksc = (const float*)(kb + TILE_N * KS8);
            constexpr unsigned NST = TILE_N / 32u;
            unsigned n_st = wave_end > b ? (wave_end - b + 31u) / 32u : 0u;
            n_st = n_st < NST ? n_st : NST;
            fp8v32 kf[2][2];
            auto ldk = [&](unsigned st, fp8v32* k) {
                const unsigned char* const kr = kb + (32u * st + frow) * KS8 + 16u * kh;
#pragma unroll
                for (int ks = 0; ks < 2; ks++) {
                    const uint4 lo = *(const uint4*)(kr + 64 * ks), hi = *(const uint4*)(kr + 64 * ks + 32);
                    k[ks] = fp8v32{(int)lo.x, (int)lo.y, (int)lo.z, (int)lo.w,
                                   (int)hi.x, (int)hi.y, (int)hi.z, (int)hi.w};
                }
            };
            auto mm = [&](const fp8v32* q, const fp8v32* k) {
                return plow_mfma_fp8_32x32(q[1], k[1], plow_mfma_fp8_32x32(q[0], k[0], (f32x16)(0.0f)));
            };
            /* row 0's sum in lanes 0-31, row 1's in lanes 32-63 after the half swap */
            auto put = [&](unsigned st, float s0, float s1) {
                const auto sw = __builtin_amdgcn_permlane32_swap(__float_as_uint(s0), __float_as_uint(s1),
                                                                 false, false);
                const float tot = __uint_as_float(sw[0]) + __uint_as_float(sw[1]);
                const unsigned pos = b + 32u * st + frow;
                if (pos < my_end) st_act<float>(&my_row[pos], tot * ksc[32u * st + frow]);
            };
            /* row 1's epilogue of subtile st-1 runs under row 0's MFMAs of st, row 0's of st under
             * row 1's */
            f32x16 a0, a1;
            float s0 = 0.0f;
            if (n_st > 0u) {
                ldk(0u, kf[0]);
                a0 = mm(qa[0], kf[0]);
            }
            if (b + TILE_N < s_hi) {
                stage(lds + (cur ^ 1u) * SLAB);
                if (b + 2u * TILE_N < s_hi) fetch(b + 2u * TILE_N);
            }
            if (n_st > 0u) {
                a1 = mm(qa[1], kf[0]);
                if (n_st > 1u) ldk(1u, kf[1]);
                s0 = head_sum(a0, wv[0]);
            }
#pragma unroll
            for (unsigned st = 1; st < NST; st++) {
                if (st < n_st) {
                    a0 = mm(qa[0], kf[st & 1u]);
                    const float s1 = head_sum(a1, wv[1]);
                    put(st - 1u, s0, s1);
                    a1 = mm(qa[1], kf[st & 1u]);
                    if (st + 1u < n_st) ldk(st + 1u, kf[(st + 1u) & 1u]);
                    s0 = head_sum(a0, wv[0]);
                }
            }
            if (n_st > 0u) put(n_st - 1u, s0, head_sum(a1, wv[1]));
            __syncthreads();
        }
    }
}
