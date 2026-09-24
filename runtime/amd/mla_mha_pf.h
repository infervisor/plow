// GLM-5.3 MLA prefill attention in the EXPANDED (MHA) form vLLM runs: per head, K = [k_nope(192) |
// k_rope(64, head-shared)], V = v(256), Q = [q_nope(192) | q_rope(64)], causal, fused bf16 output.
// Half the FLOPs of the absorbed form (QK 256 + PV 256 vs 576 + 512 per head).
//
// Transposed products so a query row is LANE-LOCAL end to end: S^T = K Q^T and O^T = V^T P^T with
// v_mfma_f32_32x32x16_bf16. The S^T accumulator holds q = lane%32, so the row max/sum are in-lane
// plus one lane^32 exchange, P is already the B operand of the PV product, and the O^T accumulator
// holds the same q, so the online rescale and the final 1/l never cross lanes. The PV k-index is
// the S^T accumulator's own kv permutation {4g..4g+3, 4g+8..4g+11}, which is exactly what two
// ds_read_b64_tr_b16 of row-major V return (4 rows x 16 columns per 16-lane group).
//
// K and V tiles are row-major [64][256] bf16 at a 280-half row stride (140 dwords = 12 mod 64
// banks: conflict-free for the 16-row b128 K reads and the 4-row x 8-dword tr16 V reads), so
// every LDS offset is a per-lane base plus an immediate. Staged global->LDS by DMA (no VGPRs),
// one row per 32-lane issue, into two stages: tile kt+1 streams while tile kt computes.
// Work item = (q-tile pair {i, n-1-i}, head): equal causal cost per item.
#pragma once

#ifndef MLA_MHA_REGSTAGE
#define MLA_MHA_REGSTAGE 0
#endif
#ifndef MLA_MHA_SCALARV
#define MLA_MHA_SCALARV 0
#endif
namespace mla_mha {
constexpr unsigned BQ = 128, BKV = 64, D = 256, DN = 192, DV = 256;
constexpr unsigned RS = D + 24;                 // row stride, halves
constexpr unsigned TILE = BKV * RS;             // halves per K or V tile
constexpr unsigned LDS_BYTES = 4 * TILE * 2;    // 2 stages x (K + V) = 140 KiB
typedef bf16_t bx4 __attribute__((ext_vector_type(4)));
}  // namespace mla_mha

__device__ __forceinline__ unsigned mla_mha_off(unsigned row, unsigned col) {
    return row * mla_mha::RS + col;
}

__device__ void d_mla_mha_pf(bf16* __restrict__ O, const bf16* __restrict__ Qn,
                             const bf16* __restrict__ Qr, const bf16* __restrict__ Kn,
                             const bf16* __restrict__ Kr, const bf16* __restrict__ V, unsigned T,
                             unsigned H, unsigned qn_rs, unsigned qn_hs, unsigned qr_rs,
                             unsigned qr_hs, unsigned kv_rs, unsigned kv_hs, unsigned kr_rs,
                             float scale, unsigned slice, unsigned nblk, bf16* lds,
                             const float* __restrict__ cosb = nullptr,
                             const float* __restrict__ sinb = nullptr,
                             const int* __restrict__ pos = nullptr) {
    using namespace mla_mha;
    const unsigned tid = threadIdx.x, lane = tid & 63u, wave = tid >> 6;
    const unsigned g = lane >> 5, r32 = lane & 31u;
    const unsigned g16 = (lane >> 4) & 1u, i16 = lane & 15u;  // tr16: dv half, lane in group
    const float sl2 = scale * 1.4426950408889634f;
    const unsigned nqt = (T + BQ - 1) / BQ, npair = (nqt + 1) / 2;
    const unsigned items = npair * H;

    // DMA: wave w stages rows 16w..16w+15 of K and V, one row per issue on lanes 0..31. Lane c
    // carries chunk c; its K source (nope or rope table) and row stride are fixed per head.
    const unsigned c = lane & 31u;
    const bool rope_lane = c >= DN / 8;
    const size_t kstride = rope_lane ? kr_rs : kv_rs;
    const bf16* kbase = nullptr;
    const bf16* vbase = nullptr;
    auto stage = [&](unsigned kt, unsigned sb) {
        bf16* Ks = lds + sb * 2u * TILE;
        bf16* Vs = Ks + TILE;
        const unsigned kv0 = kt * BKV;
        if (lane < 32u) {
#pragma unroll
            for (unsigned k = 0; k < 16; k++) {
                const unsigned row = 16u * wave + k;
                const size_t kv = min(kv0 + row, T - 1u);
#if MLA_MHA_REGSTAGE
                *(uint4*)(Ks + row * RS + c * 8u) = *(const uint4*)(kbase + kv * kstride);
                *(uint4*)(Vs + row * RS + c * 8u) = *(const uint4*)(vbase + kv * kv_rs);
#else
                cp_async16(as_glob(kbase + kv * kstride), Ks + row * RS);
                cp_async16(as_glob(vbase + kv * kv_rs), Vs + row * RS);
#endif
            }
        }
    };

    for (unsigned it = slice; it < items; it += nblk) {
        const unsigned h = it % H, p = it / H;
        kbase = rope_lane ? Kr + (c * 8u - DN) : Kn + h * kv_hs + c * 8u;
        vbase = V + h * kv_hs + c * 8u;
        for (unsigned half = 0; half < 2; half++) {
            const unsigned qt = half ? nqt - 1 - p : p;
            if (half && qt == p) break;
            const unsigned q0 = qt * BQ + wave * 32u;
            const unsigned q = q0 + r32;
            const bool qv = q < T;
            const unsigned kv_end = min(T, qt * BQ + BQ);
            const unsigned ntile = (kv_end + BKV - 1) / BKV;
            __syncthreads();  // previous item's last tile reads are done before stage 0 refills
            stage(0, 0);
            bf16x8 qf[16];
#pragma unroll
            for (unsigned s = 0; s < 16; s++) {
                const unsigned d = 16u * s + 8u * g;
                const unsigned qq = qv ? q : T - 1u;
                if (d < DN || !Qr) qf[s] = *(const bf16x8*)(Qn + (size_t)qq * qn_rs + h * qn_hs + d);
                else qf[s] = *(const bf16x8*)(Qr + (size_t)qq * qr_rs + h * qr_hs + (d - DN));
                if (d >= DN && !Qr) {
                    // Raw q_rope in Qn: GPT-J interleaved RoPE, HeadNormRope's hd=64 skip_norm
                    // arithmetic (pair m of the 64-wide strip at table index pos*32 + m).
                    const size_t tb = (size_t)pos[qq] * 32u + (d - DN) / 2u;
#pragma unroll
                    for (unsigned k = 0; k < 4; k++) {
                        const float c = cosb[tb + k], sn = sinb[tb + k];
                        const float a = (float)qf[s][2 * k], b = (float)qf[s][2 * k + 1];
                        qf[s][2 * k] = (bf16_t)(a * c - b * sn);
                        qf[s][2 * k + 1] = (bf16_t)(b * c + a * sn);
                    }
                }
            }
            f32x16 o[8];
#pragma unroll
            for (unsigned n = 0; n < 8; n++) o[n] = f32x16{};
            float mrow = -INFINITY, lrow = 0.0f;
            const unsigned wq_max = q0 + 31u;

            for (unsigned kt = 0; kt < ntile; kt++) {
                const unsigned kv0 = kt * BKV;
                cp_async_wait();
                __syncthreads();
                if (kt + 1 < ntile) stage(kt + 1, (kt + 1) & 1u);
                if (kv0 > wq_max) continue;
                const bf16* Ks = lds + (kt & 1u) * 2u * TILE;
                const bf16* Vs = Ks + TILE;

                // S^T = K Q^T, K fragments read one k-step ahead.
                f32x16 s[2] = {f32x16{}, f32x16{}};
                const bf16* kp = Ks + mla_mha_off(r32, 8u * g);
                bf16x8 ka[2][2];
                ka[0][0] = *(const bf16x8*)kp;
                ka[0][1] = *(const bf16x8*)(kp + 32u * RS);
#pragma unroll
                for (unsigned st = 0; st < 16; st++) {
                    if (st + 1 < 16) {
                        ka[(st + 1) & 1][0] = *(const bf16x8*)(kp + 16u * (st + 1));
                        ka[(st + 1) & 1][1] = *(const bf16x8*)(kp + 32u * RS + 16u * (st + 1));
                    }
                    s[0] = plow_mfma_bf16_32x32(ka[st & 1][0], qf[st], s[0]);
                    s[1] = plow_mfma_bf16_32x32(ka[st & 1][1], qf[st], s[1]);
                }
                // Scale (+ causal / tail mask on the diagonal tile only; wave-uniform branch).
                float mx = -INFINITY;
                if (kv0 + BKV - 1u > q0 || kv0 + BKV > T) {
#pragma unroll
                    for (unsigned t = 0; t < 2; t++)
#pragma unroll
                        for (unsigned i = 0; i < 16; i++) {
                            const unsigned kv = kv0 + t * 32u + 4u * g + (i & 3u) + 8u * (i >> 2);
                            const float x = (kv > q || kv >= T) ? -INFINITY : s[t][i] * sl2;
                            s[t][i] = x;
                            mx = fmaxf(mx, x);
                        }
                } else {
#pragma unroll
                    for (unsigned t = 0; t < 2; t++)
#pragma unroll
                        for (unsigned i = 0; i < 16; i++) {
                            s[t][i] *= sl2;
                            mx = fmaxf(mx, s[t][i]);
                        }
                }
                mx = fmaxf(mx, __shfl_xor(mx, 32));
                // Lazy rescale: the reference max moves only when some row grows past it by
                // more than 8 (log2), so P <= 2^8 and O is rescaled on a few tiles, not all.
                if (__any(mx > mrow + 8.0f)) {
                    const float mnew = fmaxf(mrow, mx);
                    const float corr = mnew == -INFINITY ? 1.0f : __builtin_amdgcn_exp2f(mrow - mnew);
                    lrow *= corr;
#pragma unroll
                    for (unsigned n = 0; n < 8; n++) o[n] *= corr;
                    mrow = mnew;
                }
                const float msafe = mrow == -INFINITY ? 0.0f : mrow;
                float ls = 0.0f;
                bf16x8 pf[4];
#pragma unroll
                for (unsigned t = 0; t < 2; t++)
#pragma unroll
                    for (unsigned i = 0; i < 16; i++) {
                        const float e = __builtin_amdgcn_exp2f(s[t][i] - msafe);
                        ls += e;
                        pf[t * 2 + (i >> 3)][i & 7u] = (bf16_t)e;
                    }
                lrow += ls + __shfl_xor(ls, 32);
                // O^T[dv][q] += V^T P^T. tr16 group: rows kb + (i16>>2), cols dv0 + (i16&3)*4;
                // fragments read one (k-step, dv tile) ahead.
                const bf16* vp = Vs + mla_mha_off(4u * g + (i16 >> 2), g16 * 16u + (i16 & 3u) * 4u);
                auto vfrag = [&](unsigned st, unsigned n) {
                    const bx4 lo = mla_pf_ds_read_tr16(vp + st * 16u * RS + n * 32u);
                    const bx4 hi = mla_pf_ds_read_tr16(vp + (st * 16u + 8u) * RS + n * 32u);
                    return bf16x8{lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]};
                };
                bf16x8 va[2];
                va[0] = vfrag(0, 0);
#pragma unroll
                for (unsigned x = 0; x < 32; x++) {
                    const unsigned st = x >> 3, n = x & 7u;
                    if (x + 1 < 32) va[(x + 1) & 1] = vfrag((x + 1) >> 3, (x + 1) & 7u);
                    o[n] = plow_mfma_bf16_32x32(va[x & 1], pf[st], o[n]);
                }
            }
            if (qv) {
                const float inv = lrow > 0.0f ? 1.0f / lrow : 0.0f;
                bf16* orow = O + ((size_t)q * H + h) * DV;
#pragma unroll
                for (unsigned n = 0; n < 8; n++)
#pragma unroll
                    for (unsigned i4 = 0; i4 < 4; i4++) {
                        bx4 w;
#pragma unroll
                        for (unsigned c = 0; c < 4; c++) w[c] = (bf16_t)(o[n][i4 * 4 + c] * inv);
                        *(bx4*)(orow + n * 32u + 4u * g + 8u * i4) = w;
                    }
            }
        }
    }
}
