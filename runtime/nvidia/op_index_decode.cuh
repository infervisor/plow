/* op_index_decode.cuh -- DeepSeek-V4.1 indexer, one decode token per slot.
 *
 * Reference: inference/model.py Indexer.forward at start_pos > 0. Per slot b the query sees
 * len(b) = (pos[b] + 1) / ratio compressed positions:
 *     score[b][t] = sum_h w[b][h] * relu(q[b][h] . k[b][t])          t < len(b)
 *     idx[b]      = top-min(512, len) t by score, ascending t, -1 padded
 * w is weights_proj's bf16 output; wscale = softmax_scale * n_heads^-0.5 is applied here (the prefill
 * op folds it into its epilogue; same expression by distributivity). At TP the score is rank-partial (heads are
 * split), so scoring and selection are two functions with the all-reduce between them.
 *
 * Score: memory bound on the keys (256 B per position). Keys are the MMA M dimension, the rank's
 * heads (8 at TP4, 32 at TP1) are N: S^T[t][h] = K[t][0..128) . Q^T. mma.sync m16n8k16 reads its
 * A/B fragments straight from global: a dot product is invariant under one permutation of k applied
 * to both operands, so each lane loads 16 contiguous bytes of its key row and of its query row and
 * both sides agree on which logical k each register pair holds. Slots with len <= 512 skip the
 * score (selection takes every position).
 *
 * Select: one CTA per slot, narrowing the candidates by 256-bucket histogram levels until the 512th
 * largest is isolated; positions are written in ascending order via two block scans, so the output
 * is deterministic.
 */
#pragma once
#include <cuda_bf16.h>
#include <stdint.h>

namespace plow_idd {

constexpr unsigned DI = 128, TOPK = 512, UNIT = 64; /* UNIT = keys per warp work item */

__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__host__ __device__ __forceinline__ unsigned len_of(int pos, unsigned ratio) { return ((unsigned)pos + 1u) / ratio; }
__device__ __forceinline__ unsigned units_of(unsigned len) { return len > TOPK ? (len + UNIT - 1) / UNIT : 0u; }
__device__ __forceinline__ unsigned okey(float s) {
    const unsigned u = __float_as_uint(s);
    return u ^ ((u >> 31) ? 0xffffffffu : 0x80000000u);
}
__device__ __forceinline__ uint4 ldg_nc(const void* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

/* NT = heads / 8 */
template <int NT>
__device__ void index_score_decode(float* __restrict__ score, const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ w, float wscale,
                                   const __nv_bfloat16* __restrict__ k, const int* __restrict__ pos, unsigned B, unsigned cap,
                                   unsigned ratio, unsigned slice, unsigned nblk) {
    constexpr unsigned HI = NT * 8;
    const unsigned lane = threadIdx.x & 31, g = lane >> 2, qd = lane & 3;
    const unsigned nw = blockDim.x >> 5, gw = slice * nw + (threadIdx.x >> 5), NW = nblk * nw;
    /* warp-cooperative unit count (a serial walk over pos[] is ~2.5 us at B=64) */
    unsigned U = 0;
    for (unsigned b0 = 0; b0 < B; b0 += 32) {
        unsigned n = b0 + lane < B ? units_of(len_of(pos[b0 + lane], ratio)) : 0u;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) n += __shfl_xor_sync(0xffffffffu, n, o);
        U += n;
    }
    /* contiguous unit ranges per warp: the query fragments reload only when the slot changes */
    const unsigned u0 = (unsigned)((unsigned long long)U * gw / NW), u1 = (unsigned)((unsigned long long)U * (gw + 1) / NW);
    if (u0 >= u1) return;
    /* slot holding unit u0: inclusive unit prefix per lane, first lane past u0 */
    unsigned b = 0, base = 0;
    for (unsigned b0 = 0; b0 < B; b0 += 32) {
        const unsigned n = b0 + lane < B ? units_of(len_of(pos[b0 + lane], ratio)) : 0u;
        unsigned incl = n;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const unsigned x = __shfl_up_sync(0xffffffffu, incl, o);
            if (lane >= (unsigned)o) incl += x;
        }
        const unsigned hit = __ballot_sync(0xffffffffu, base + incl > u0);
        if (hit) {
            const unsigned l = __ffs(hit) - 1;
            b = b0 + l;
            base += __shfl_sync(0xffffffffu, incl - n, l);
            break;
        }
        base += __shfl_sync(0xffffffffu, incl, 31);
    }
    unsigned len = len_of(pos[b], ratio), nu = units_of(len);
    int loaded = -1;
    uint32_t bq[NT][8][2];
    float wv[NT][2];
    for (unsigned u = u0; u < u1; u++) {
        while (u >= base + nu) base += nu, b++, len = len_of(pos[b], ratio), nu = units_of(len);
        if ((int)b != loaded) {
            loaded = (int)b;
#pragma unroll
            for (int nt = 0; nt < NT; nt++) {
                const __nv_bfloat16* qr = q + ((size_t)b * HI + nt * 8 + g) * DI + qd * 8;
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const uint4 v = *reinterpret_cast<const uint4*>(qr + j * 32);
                    bq[nt][2 * j][0] = v.x, bq[nt][2 * j][1] = v.y, bq[nt][2 * j + 1][0] = v.z, bq[nt][2 * j + 1][1] = v.w;
                }
                wv[nt][0] = __bfloat162float(w[b * HI + nt * 8 + qd * 2]) * wscale;
                wv[nt][1] = __bfloat162float(w[b * HI + nt * 8 + qd * 2 + 1]) * wscale;
            }
        }
        const __nv_bfloat16* kb = k + (size_t)b * cap * DI;
        const unsigned t0 = (u - base) * UNIT;
#pragma unroll
        for (unsigned half = 0; half < UNIT / 32; half++) {
            uint4 kr[2][2][4]; /* [tile][row g / g+8][j] */
#pragma unroll
            for (int ti = 0; ti < 2; ti++)
#pragma unroll
                for (int rr = 0; rr < 2; rr++) {
                    const unsigned t = t0 + half * 32 + ti * 16 + rr * 8 + g;
                    const __nv_bfloat16* kr_ = kb + (size_t)(t < len ? t : 0) * DI + qd * 8;
#pragma unroll
                    for (int j = 0; j < 4; j++) kr[ti][rr][j] = t < len ? ldg_nc(kr_ + j * 32) : make_uint4(0, 0, 0, 0);
                }
#pragma unroll
            for (int ti = 0; ti < 2; ti++) {
                float c[NT][4];
#pragma unroll
                for (int nt = 0; nt < NT; nt++) c[nt][0] = c[nt][1] = c[nt][2] = c[nt][3] = 0.f;
#pragma unroll
                for (int j = 0; j < 4; j++)
#pragma unroll
                    for (int nt = 0; nt < NT; nt++) {
                        mma16816(c[nt], kr[ti][0][j].x, kr[ti][1][j].x, kr[ti][0][j].y, kr[ti][1][j].y, bq[nt][2 * j][0], bq[nt][2 * j][1]);
                        mma16816(c[nt], kr[ti][0][j].z, kr[ti][1][j].z, kr[ti][0][j].w, kr[ti][1][j].w, bq[nt][2 * j + 1][0],
                                 bq[nt][2 * j + 1][1]);
                    }
                float s0 = 0.f, s1 = 0.f;
#pragma unroll
                for (int nt = 0; nt < NT; nt++) {
                    s0 += fmaxf(c[nt][0], 0.f) * wv[nt][0] + fmaxf(c[nt][1], 0.f) * wv[nt][1];
                    s1 += fmaxf(c[nt][2], 0.f) * wv[nt][0] + fmaxf(c[nt][3], 0.f) * wv[nt][1];
                }
                s0 += __shfl_xor_sync(0xffffffffu, s0, 1);
                s0 += __shfl_xor_sync(0xffffffffu, s0, 2);
                s1 += __shfl_xor_sync(0xffffffffu, s1, 1);
                s1 += __shfl_xor_sync(0xffffffffu, s1, 2);
                const unsigned t = t0 + half * 32 + ti * 16 + g;
                float* sb = score + (size_t)b * cap;
                if (qd == 0 && t < len) sb[t] = s0;
                if (qd == 1 && t + 8 < len) sb[t + 8] = s1;
            }
        }
    }
}

/* inclusive block scan of one value per thread (blockDim.x <= 1024), total in *tot */
__device__ __forceinline__ unsigned block_scan_incl(unsigned v, unsigned* sm, unsigned* tot) {
    const unsigned lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const unsigned n = __shfl_up_sync(0xffffffffu, v, o);
        if (lane >= (unsigned)o) v += n;
    }
    if (lane == 31) sm[wid] = v;
    __syncthreads();
    if (wid == 0) {
        unsigned x = lane < nw ? sm[lane] : 0u;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const unsigned n = __shfl_up_sync(0xffffffffu, x, o);
            if (lane >= (unsigned)o) x += n;
        }
        if (lane < nw) sm[lane] = x;
    }
    __syncthreads();
    const unsigned r = v + (wid ? sm[wid - 1] : 0u);
    *tot = sm[nw - 1];
    __syncthreads();
    return r;
}

} // namespace plow_idd

/* score [B][cap] f32; q [B][HI][128] bf16; w [B][HI] bf16 (x wscale); k [B][cap][128]
 * bf16; pos [B] i32. HI in {8, 16, 32}. Only slots with len > 512 are written. */
__device__ void d_index_score_decode(float* score, const __nv_bfloat16* q, const __nv_bfloat16* w, float wscale, const __nv_bfloat16* k, const int* pos,
                                     unsigned B, unsigned HI, unsigned cap, unsigned ratio, unsigned slice, unsigned nblk) {
    if (HI == 8) plow_idd::index_score_decode<1>(score, q, w, wscale, k, pos, B, cap, ratio, slice, nblk);
    else if (HI == 16) plow_idd::index_score_decode<2>(score, q, w, wscale, k, pos, B, cap, ratio, slice, nblk);
    else if (HI == 32) plow_idd::index_score_decode<4>(score, q, w, wscale, k, pos, B, cap, ratio, slice, nblk);
    else __trap();
}

namespace plow_idd {

/* block-wide (min, max) over 256..1024 threads; red holds >= 64 floats */
__device__ __forceinline__ void block_minmax(float& mn, float& mx, float* red) {
    const unsigned lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        mn = fminf(mn, __shfl_xor_sync(0xffffffffu, mn, o));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    }
    if (lane == 0) red[wid] = mn, red[32 + wid] = mx;
    __syncthreads();
    mn = red[0], mx = red[32];
    for (unsigned i = 1; i < nw; i++) mn = fminf(mn, red[i]), mx = fmaxf(mx, red[32 + i]);
    __syncthreads();
}

/* warp 0 walks hist[256] from the top: sel[0] = digit whose cumulative count reaches need,
 * sel[1] = how many of that digit are still needed, sel[2] = the digit's count */
__device__ __forceinline__ void pick_digit(const unsigned* hist, unsigned need, unsigned* sel) {
    const unsigned tid = threadIdx.x;
    if (tid < 32) {
        unsigned c[8], s = 0;
#pragma unroll
        for (int i = 0; i < 8; i++) c[i] = hist[255 - tid * 8 - i], s += c[i];
        unsigned incl = s;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const unsigned n = __shfl_up_sync(0xffffffffu, incl, o);
            if (tid >= (unsigned)o) incl += n;
        }
        unsigned above = incl - s;
        if (above < need && incl >= need) {
#pragma unroll
            for (int i = 0; i < 8; i++) {
                if (above + c[i] >= need) {
                    sel[0] = 255 - tid * 8 - i;
                    sel[1] = need - above;
                    sel[2] = c[i];
                    break;
                }
                above += c[i];
            }
        }
    }
    __syncthreads();
}

__device__ __forceinline__ unsigned lin_bucket(float v, float lo, float inv) { return min((unsigned)((v - lo) * inv), 255u); }

} // namespace plow_idd

/* idx [B][512] i32 = top-min(512, len) positions of score[b][0..len), ascending, -1 padded; ties
 * go to the lowest position. cap % 4 == 0 (rows are read as float4). smem: arena of smem_words u32, >= 384. A slot whose len fits
 * (384 + len words) selects in smem by value-linear 256-bucket levels over the remaining
 * candidates' [min, max] (scores bunch into a few exponents, so radix digits on the float bits
 * serialize the smem atomics: 38 us at len 16.5k vs this path's); otherwise an 8-bit radix
 * over the global scores. */
__device__ void d_index_select_decode(int* __restrict__ idx, const float* __restrict__ score, const int* __restrict__ pos, unsigned B,
                                      unsigned cap, unsigned ratio, unsigned slice, unsigned nblk, unsigned* smem, unsigned smem_words) {
    using namespace plow_idd;
    unsigned* hist = smem;       /* [256] */
    unsigned* scan = smem + 256; /* [32] */
    unsigned* sel = smem + 288;  /* [4] */
    float* red = (float*)(smem + 320); /* [64] */
    float* vals = (float*)(smem + 384);
    const unsigned tid = threadIdx.x, nt = blockDim.x;
    for (unsigned b = slice; b < B; b += nblk) {
        const unsigned len = len_of(pos[b], ratio);
        int* ib = idx + (size_t)b * TOPK;
        if (len <= TOPK) {
            for (unsigned i = tid; i < TOPK; i += nt) ib[i] = i < len ? (int)i : -1;
            continue;
        }
        const float* sb = score + (size_t)b * cap;
        const unsigned chunk = (len + nt - 1) / nt, lo_i = min(tid * chunk, len), hi_i = min(lo_i + chunk, len);
        unsigned tot;
        __syncthreads();
        if (384 + len <= smem_words) {
            /* vals[i]: the score while a candidate, +inf once taken, -inf once out (scores are
             * finite). Loops batch 8 independent smem/global loads: one-at-a-time they are
             * latency-serialized at ~150 cycles per element. */
            constexpr unsigned BT = 8;
            const unsigned len4 = len / 4;
            for (unsigned i0 = tid; i0 < len4; i0 += nt * BT) {
                float4 v[BT];
#pragma unroll
                for (unsigned u = 0; u < BT; u++)
                    if (i0 + u * nt < len4) v[u] = __ldcg(reinterpret_cast<const float4*>(sb) + i0 + u * nt);
#pragma unroll
                for (unsigned u = 0; u < BT; u++)
                    if (i0 + u * nt < len4) reinterpret_cast<float4*>(vals)[i0 + u * nt] = v[u];
            }
            for (unsigned i = len4 * 4 + tid; i < len; i += nt) vals[i] = __ldcg(sb + i);
            __syncthreads();
            unsigned need = TOPK, d = 0;
            float lo = 0.f, inv = 0.f;
            bool rule = false, take_all = false;
            for (int level = 0; level < 8; level++) {
                /* resolve the previous level's buckets, (min, max) of what is left */
                float mn = INFINITY, mx = -INFINITY;
                for (unsigned i0 = tid; i0 < len; i0 += nt * BT) {
                    float v[BT];
#pragma unroll
                    for (unsigned u = 0; u < BT; u++) v[u] = i0 + u * nt < len ? vals[i0 + u * nt] : -INFINITY;
#pragma unroll
                    for (unsigned u = 0; u < BT; u++) {
                        if (!isfinite(v[u])) continue;
                        if (rule) {
                            const unsigned bk = lin_bucket(v[u], lo, inv);
                            if (bk != d) {
                                vals[i0 + u * nt] = bk > d ? INFINITY : -INFINITY;
                                continue;
                            }
                        }
                        mn = fminf(mn, v[u]), mx = fmaxf(mx, v[u]);
                    }
                }
                block_minmax(mn, mx, red);
                rule = false;
                if (mx == mn) break; /* all remaining are equal: the lowest positions win */
                lo = mn, inv = 256.f / (mx - mn);
                for (unsigned i = tid; i < 256; i += nt) hist[i] = 0;
                __syncthreads();
                for (unsigned i0 = tid; i0 < len; i0 += nt * BT) {
                    float v[BT];
#pragma unroll
                    for (unsigned u = 0; u < BT; u++) v[u] = i0 + u * nt < len ? vals[i0 + u * nt] : -INFINITY;
#pragma unroll
                    for (unsigned u = 0; u < BT; u++)
                        if (isfinite(v[u])) atomicAdd(hist + lin_bucket(v[u], lo, inv), 1u);
                }
                __syncthreads();
                pick_digit(hist, need, sel);
                d = sel[0], need = sel[1], rule = true, take_all = sel[2] == need;
                __syncthreads();
                if (take_all) break;
            }
            /* class: 1 = sure (taken, above the last digit, or in it when it is taken whole),
             * 2 = group (the remaining candidates; the `need` lowest positions are taken), 0 = out */
            auto cls = [&](float v) -> unsigned {
                if (v == INFINITY) return 1u;
                if (!isfinite(v)) return 0u;
                if (!rule) return 2u;
                const unsigned bk = lin_bucket(v, lo, inv);
                return bk > d || (take_all && bk == d) ? 1u : bk == d ? 2u : 0u;
            };
            unsigned sure = 0, grp = 0;
            for (unsigned j0 = lo_i; j0 < hi_i; j0 += BT) {
                float v[BT];
#pragma unroll
                for (unsigned u = 0; u < BT; u++) v[u] = j0 + u < hi_i ? vals[j0 + u] : -INFINITY;
#pragma unroll
                for (unsigned u = 0; u < BT; u++) {
                    const unsigned c = cls(v[u]);
                    sure += c == 1u, grp += c == 2u;
                }
            }
            const unsigned g_before = block_scan_incl(grp, scan, &tot) - grp;
            const unsigned take_g = g_before >= need ? 0u : min(grp, need - g_before);
            unsigned o = block_scan_incl(sure + take_g, scan, &tot) - (sure + take_g), tg = take_g;
            for (unsigned j0 = lo_i; j0 < hi_i; j0 += BT) {
                float v[BT];
#pragma unroll
                for (unsigned u = 0; u < BT; u++) v[u] = j0 + u < hi_i ? vals[j0 + u] : -INFINITY;
#pragma unroll
                for (unsigned u = 0; u < BT; u++) {
                    const unsigned c = cls(v[u]);
                    const bool t = c == 1u || (c == 2u && tg);
                    if (c == 2u && tg) tg--;
                    if (t) ib[o++] = (int)(j0 + u);
                }
            }
            continue;
        }
        unsigned prefix = 0, mask = 0, need = TOPK;
        for (int shift = 24; shift >= 0; shift -= 8) {
            for (unsigned i = tid; i < 256; i += nt) hist[i] = 0;
            __syncthreads();
            for (unsigned i = tid; i < len; i += nt) {
                const unsigned u = okey(__ldcg(sb + i));
                if ((u & mask) == prefix) atomicAdd(hist + ((u >> shift) & 255u), 1u);
            }
            __syncthreads();
            pick_digit(hist, need, sel);
            prefix |= sel[0] << shift;
            mask |= 255u << shift;
            need = sel[1];
            __syncthreads();
        }
        /* prefix = T, need = how many == T to take (lowest positions first) */
        unsigned gt = 0, eq = 0;
        for (unsigned i = lo_i; i < hi_i; i++) {
            const unsigned u = okey(__ldcg(sb + i));
            gt += u > prefix;
            eq += u == prefix;
        }
        const unsigned eq_before = block_scan_incl(eq, scan, &tot) - eq;
        const unsigned take_eq = eq_before >= need ? 0u : min(eq, need - eq_before);
        unsigned o = block_scan_incl(gt + take_eq, scan, &tot) - (gt + take_eq), te = take_eq;
        for (unsigned i = lo_i; i < hi_i; i++) {
            const unsigned u = okey(__ldcg(sb + i));
            if (u > prefix || (u == prefix && te)) {
                if (u == prefix) te--;
                ib[o++] = (int)i;
            }
        }
    }
}
