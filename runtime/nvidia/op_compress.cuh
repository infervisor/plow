/* op_compress.cuh -- DeepSeek-V4.1's KV compressor tail on the warp32 interpreters. Same operands and
 * semantics as runtime/amd/op_compress.h (read its notes), restricted to what V4.1 emits:
 *
 * PLOW_DOP_COMPRESS_POOL (194), arm i7 == 2 only: per-channel softmax pool over coff*ratio slots,
 *   bf16 round, RMSNorm(gamma), bf16 round, STOP (model.py Compressor.forward). V4's rope/quant and
 *   Hadamard epilogues (i7 0/1) trap.
 * PLOW_DOP_COMPRESS_ROPE_QUANT (199): interleaved rope of the last rd channels at (row_base+r)*ratio,
 *   then a fake quant per qblk block (kernel.py fp4_act_quant / act_quant, inplace=True). One thread
 *   per block, 16 B loads; needs qblk % 8 == 0, qblk <= 32, (d - rd) % 8 == 0 and (rd/2) % 4 == 0,
 *   which all three V4.1 call sites meet.
 * PLOW_DOP_ROPE_INVERSE_O (195): conjugate rotation of the last rd dims of each head, in place.
 *
 * The fp4 rounding is nearest-even (the reference's cast); the AMD ladder rounds exact ties away
 * from zero, so the two differ on exact midpoints only.
 */
#pragma once
#include <cuda_bf16.h>
#include <stdint.h>

#define PLOW_CMP_Q_FP8_POW2 0u
#define PLOW_CMP_Q_FP4_POW2 1u
#define PLOW_CMP_Q_FP4_E4M3 2u

namespace plow_cmp {
__device__ __forceinline__ float bf(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }
__device__ __forceinline__ uint16_t to_bf(float f) {
    const __nv_bfloat16 v = __float2bfloat16_rn(f);
    return *(const uint16_t*)&v;
}
__device__ __forceinline__ float rbf(float f) { return bf(to_bf(f)); }

/* 2^ceil(log2(t)), t > 0 finite (kernel.py fast_round_scale) */
__device__ __forceinline__ float pow2_ceil(float t) {
    const uint32_t u = __float_as_uint(t);
    const int e = (int)(u >> 23) - 127;
    if (e == -127) return __uint_as_float(1u << 23);  // subnormal -> 2^-126
    return __uint_as_float((uint32_t)(e + ((u & 0x7fffffu) != 0u) + 127) << 23);
}
/* nearest e4m3 (ties to even), saturating at 448 */
__device__ __forceinline__ float round_e4m3(float y) {
    const float a = fabsf(y);
    if (a == 0.f) return y;
    const int e = (int)((__float_as_uint(a) >> 23) & 0xffu) - 127;
    const float quantum = __uint_as_float((uint32_t)((e < -6 ? -6 : e) - 3 + 127) << 23);
    const float q = fminf(rintf(a / quantum) * quantum, 448.f);
    return y < 0.f ? -q : q;
}
/* nearest e2m1 (ties to even), |y| <= 6: 0 .5 1 1.5 2 3 4 6 */
__device__ __forceinline__ float round_e2m1(float y) {
    const float a = fabsf(y);
    const float q = a <= 0.25f ? 0.f : a < 0.75f ? 0.5f : a <= 1.25f ? 1.f : a < 1.75f ? 1.5f
                  : a <= 2.5f  ? 2.f : a < 3.5f  ? 3.f  : a <= 5.f   ? 4.f : 6.f;
    return y < 0.f ? -q : q;
}
__device__ __forceinline__ float block_scale(float amax, unsigned qmode) {
    if (qmode == PLOW_CMP_Q_FP8_POW2) return pow2_ceil(fmaxf(amax, 1e-4f) * (1.0f / 448.0f));
    if (qmode == PLOW_CMP_Q_FP4_POW2) return pow2_ceil(fmaxf(amax, 7.052966328760454e-38f) * (1.0f / 6.0f));
    return round_e4m3(fmaxf(amax, 0.01171875f) / 6.0f);
}
__device__ __forceinline__ float quant_rt(float x, float s, unsigned qmode) {
    if (qmode == PLOW_CMP_Q_FP8_POW2) return rbf(round_e4m3(fminf(fmaxf(x / s, -448.f), 448.f)) * s);
    return rbf(round_e2m1(fminf(fmaxf(x / s, -6.f), 6.f)) * s);
}
__device__ __forceinline__ void unpack8(uint4 v, float* o) {
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int q = 0; q < 4; q++) {
        o[2 * q] = __uint_as_float(w[q] << 16);
        o[2 * q + 1] = __uint_as_float(w[q] & 0xffff0000u);
    }
}
}  // namespace plow_cmp

/* t0=out t1=kv t2=score t3=ape t4=gamma t7=pos · i0=n_pools i1=ratio i2=coff i3=d i5=kv/score bytes (4: f32)
 * i6=out_base f0=eps. arena: d + 32 floats. */
template <typename In>
__device__ __forceinline__ void d_compress_pool_norm(__nv_bfloat16* __restrict__ out, const In* __restrict__ kv,
                                                     const In* __restrict__ score, const float* __restrict__ ape,
                                                     const __nv_bfloat16* __restrict__ gamma, unsigned n_pools, unsigned ratio,
                                                     unsigned coff, unsigned d, float eps, unsigned out_base, unsigned slice,
                                                     unsigned nblk, float* __restrict__ arena, unsigned arena_floats,
                                                     const int* __restrict__ pos) {
    using namespace plow_cmp;
    if (d + 32u > arena_floats) __trap();
    if (pos != nullptr && ((pos[0] + 1) % (int)ratio) != 0) return;
    const unsigned pbase = pos != nullptr ? (unsigned)pos[0] / ratio : out_base;
    const unsigned nslot = coff * ratio, kstr = coff * d;
    const auto ld = [](const In* p, size_t i) {
        if constexpr (sizeof(In) == 4) return reinterpret_cast<const float*>(p)[i];
        else return bf(reinterpret_cast<const uint16_t*>(p)[i]);
    };
    const uint16_t* g16 = reinterpret_cast<const uint16_t*>(gamma);
    float* lds = arena;
    float* red = arena + d;
    const unsigned ln = threadIdx.x & 31u, wv = threadIdx.x >> 5, warps = blockDim.x >> 5;
    for (unsigned pool = slice; pool < n_pools; pool += nblk) {
        for (unsigned c = threadIdx.x; c < d; c += blockDim.x) {
            float mx = -3.0e38f;
            for (unsigned s = 0; s < nslot; s++) {
                const unsigned half = s / ratio, r = s % ratio;
                const int row = pos != nullptr ? (int)s : (int)(pool * ratio + r) - (int)((coff == 2 && half == 0) ? ratio : 0);
                if (row < 0) continue;
                mx = fmaxf(mx, ld(score, (size_t)row * kstr + half * d + c) + (ape ? ape[(size_t)r * kstr + half * d + c] : 0.f));
            }
            float den = 0.f, acc = 0.f;
            for (unsigned s = 0; s < nslot; s++) {
                const unsigned half = s / ratio, r = s % ratio;
                const int row = pos != nullptr ? (int)s : (int)(pool * ratio + r) - (int)((coff == 2 && half == 0) ? ratio : 0);
                if (row < 0) continue;
                const size_t o = (size_t)row * kstr + half * d + c;
                const float p = expf(ld(score, o) + (ape ? ape[(size_t)r * kstr + half * d + c] : 0.f) - mx);
                den += p;
                acc += ld(kv, o) * p;
            }
            lds[c] = rbf(den > 0.f ? acc / den : 0.f);
        }
        float sq = 0.f;
        for (unsigned c = threadIdx.x; c < d; c += blockDim.x) sq += lds[c] * lds[c];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
        if (ln == 0) red[wv] = sq;
        __syncthreads();
        float tot = 0.f;
        for (unsigned w = 0; w < warps; w++) tot += red[w];
        const float inv = rsqrtf(tot / (float)d + eps);
        uint16_t* orow = reinterpret_cast<uint16_t*>(out) + (size_t)(pbase + pool) * d;
        for (unsigned c = threadIdx.x; c < d; c += blockDim.x) orow[c] = to_bf(lds[c] * inv * bf(g16[c]));
        __syncthreads();  // lds / red reused by the next pool
    }
}

/* V4.1 Compressor decode step (model.py Compressor.forward, start_pos > 0) for B slots, one CTA per
 * slot: state[b][pos%ratio] = (kv[b], score[b]); when (pos[b]+1) % ratio == 0, latent[b] =
 * RMSNorm(bf16(sum_r softmax_r(score_state) * kv_state)) * gamma. Slots whose group is still filling
 * leave latent[b] untouched (every consumer gates on the same predicate). State is f32
 * [B][ratio][d]; kv/score f32 [B][d]; latent bf16 [B][d]. arena: d + 32 floats.
 * seed (prefill of one slot, pos = kvlen): kv/score are the chunk's [T][d] rows; the incomplete tail
 * group, rows [L - L%ratio, L), goes to state slots [0, L%ratio). No pooling. */
__device__ __forceinline__ void d_compress_decode_step(__nv_bfloat16* __restrict__ latent, float* __restrict__ st_kv,
                                                       float* __restrict__ st_sc, const float* __restrict__ kv,
                                                       const float* __restrict__ score, const __nv_bfloat16* __restrict__ gamma,
                                                       const int* __restrict__ pos, unsigned B, unsigned ratio, unsigned d, float eps,
                                                       unsigned slice, unsigned nblk, float* arena, unsigned seed = 0) {
    using namespace plow_cmp;
    if (seed) {
        const unsigned L = (unsigned)pos[0], tail = L % ratio, r0 = L - tail;
        for (size_t e = (size_t)slice * blockDim.x + threadIdx.x; e < (size_t)tail * d; e += (size_t)nblk * blockDim.x) {
            st_kv[e] = kv[(size_t)r0 * d + e];
            st_sc[e] = score[(size_t)r0 * d + e];
        }
        return;
    }
    float* lds = arena;
    float* red = arena + d;
    const unsigned ln = threadIdx.x & 31u, wv = threadIdx.x >> 5, warps = blockDim.x >> 5;
    const uint16_t* g16 = reinterpret_cast<const uint16_t*>(gamma);
    for (unsigned b = slice; b < B; b += nblk) {
        const int p = pos[b];
        const unsigned slot = (unsigned)p % ratio;
        float* skv = st_kv + (size_t)b * ratio * d;
        float* ssc = st_sc + (size_t)b * ratio * d;
        const bool fire = ((p + 1) % (int)ratio) == 0;
        float sq = 0.f;
        for (unsigned c = threadIdx.x; c < d; c += blockDim.x) {
            const float k = kv[(size_t)b * d + c], sc = score[(size_t)b * d + c];
            skv[(size_t)slot * d + c] = k;
            ssc[(size_t)slot * d + c] = sc;
            if (!fire) continue;
            float mx = sc;
            for (unsigned r = 0; r < ratio; r++)
                if (r != slot) mx = fmaxf(mx, ssc[(size_t)r * d + c]);
            float den = 0.f, acc = 0.f;
            for (unsigned r = 0; r < ratio; r++) {
                const float e = expf((r == slot ? sc : ssc[(size_t)r * d + c]) - mx);
                den += e;
                acc += (r == slot ? k : skv[(size_t)r * d + c]) * e;
            }
            const float v = rbf(acc / den);
            lds[c] = v;
            sq += v * v;
        }
        if (!fire) continue; /* uniform per CTA */
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
        if (ln == 0) red[wv] = sq;
        __syncthreads();
        float tot = 0.f;
        for (unsigned w = 0; w < warps; w++) tot += red[w];
        const float inv = rsqrtf(tot / (float)d + eps);
        uint16_t* orow = reinterpret_cast<uint16_t*>(latent) + (size_t)b * d;
        for (unsigned c = threadIdx.x; c < d; c += blockDim.x) orow[c] = to_bf(lds[c] * inv * bf(g16[c]));
        __syncthreads();
    }
}

/* t0=out t1=src t2=cosb t3=sinb t4=pos · i0=n_rows i1=d i2=rd i3=qblk i4=ratio i5=row_base i6=qmode i7=n_head
 * j1 bit31=batched, j0=slot_stride. Batched decode: row r is slot r at pos[r]; it runs only when its group
 * completes ((pos[r]+1) % ratio == 0), ropes at the group's first position, reads src row r and
 * writes out row r*slot_stride + pos[r]/ratio (slot_stride 0: out row r, e.g. the indexer query).
 * j1 & 0x7fffffff = ring_mask != 0 wraps that row index (the window ring: ratio 1, stride 128, mask 127).
 * t5=kvlen (prefill SEED of one slot): row r is a chunk row at pos[r]; rows at or past kvlen[0], and on
 * a ring rows older than kvlen[0] - (ring_mask + 1), are skipped; out row = pos[r]/ratio (wrapped). */
__device__ __forceinline__ void d_compress_rope_quant(__nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ src,
                                                      const float* __restrict__ cosb, const float* __restrict__ sinb, unsigned n_rows,
                                                      unsigned d, unsigned rd, unsigned qblk, unsigned ratio, unsigned row_base,
                                                      unsigned qmode, unsigned slice, unsigned nblk, const int* __restrict__ pos,
                                                      unsigned n_head, unsigned batched = 0, unsigned slot_stride = 0,
                                                      unsigned ring_mask = 0, const int* __restrict__ kvlen = nullptr) {
    using namespace plow_cmp;
    const unsigned c_rope0 = d - rd;
    if (d % qblk || qblk % 8u || qblk > 32u || c_rope0 % 8u || (rd / 2u) % 4u) __trap();
    if (!batched && pos != nullptr && ((pos[0] + 1) % (int)ratio) != 0) return;
    const unsigned rbase = batched ? 0u : pos != nullptr ? (unsigned)pos[0] / ratio : row_base;
    const unsigned nb = d / qblk;
    const size_t total = (size_t)n_rows * n_head * nb;
    const uint16_t* s16 = reinterpret_cast<const uint16_t*>(src);
    uint16_t* o16 = reinterpret_cast<uint16_t*>(out);
    for (size_t w = (size_t)slice * blockDim.x + threadIdx.x; w < total; w += (size_t)nblk * blockDim.x) {
        const unsigned it = (unsigned)(w / nb), c0 = (unsigned)(w % nb) * qblk, r = it / n_head;
        size_t off = ((size_t)rbase * n_head + it) * d, ooff = off;
        size_t tb = (size_t)((rbase + r) * ratio) * (rd / 2u);
        if (batched) {
            const int p = pos[r];
            if ((p + 1) % (int)ratio) continue;
            const unsigned g = (unsigned)p / ratio;
            const unsigned gw = ring_mask ? (g & ring_mask) : g;
            if (kvlen) {
                if (p >= kvlen[0] || (ring_mask && p + (int)ring_mask + 1 < kvlen[0])) continue;
                ooff = ((size_t)gw * n_head + it % n_head) * d;
            } else {
                ooff = ((slot_stride ? (size_t)r * slot_stride + gw : (size_t)r) * n_head + it % n_head) * d;
            }
            tb = (size_t)(g * ratio) * (rd / 2u);
        }
        float v[32];
        float amax = 0.f;
#pragma unroll
        for (unsigned i0 = 0; i0 < 32u; i0 += 8u) {
            if (i0 >= qblk) break;
            const unsigned c8 = c0 + i0;
            float x[8];
            unpack8(*reinterpret_cast<const uint4*>(s16 + off + c8), x);
            if (c8 >= c_rope0) {
                const size_t mb = tb + ((c8 - c_rope0) >> 1);
                const float4 cv = *reinterpret_cast<const float4*>(cosb + mb);
                const float4 sv = *reinterpret_cast<const float4*>(sinb + mb);
                const float cs[4] = {cv.x, cv.y, cv.z, cv.w}, sn[4] = {sv.x, sv.y, sv.z, sv.w};
#pragma unroll
                for (int p = 0; p < 4; p++) {
                    const float a = x[2 * p], b = x[2 * p + 1];
                    x[2 * p] = rbf(a * cs[p] - b * sn[p]);
                    x[2 * p + 1] = rbf(a * sn[p] + b * cs[p]);
                }
            }
#pragma unroll
            for (int j = 0; j < 8; j++) {
                v[i0 + j] = x[j];
                amax = fmaxf(amax, fabsf(x[j]));
            }
        }
        const float s = block_scale(amax, qmode);
#pragma unroll
        for (unsigned i0 = 0; i0 < 32u; i0 += 8u) {
            if (i0 >= qblk) break;
            uint32_t pk[4];
#pragma unroll
            for (int q = 0; q < 4; q++)
                pk[q] = (uint32_t)to_bf(quant_rt(v[i0 + 2 * q], s, qmode)) | ((uint32_t)to_bf(quant_rt(v[i0 + 2 * q + 1], s, qmode)) << 16);
            *reinterpret_cast<uint4*>(o16 + ooff + c0 + i0) = make_uint4(pk[0], pk[1], pk[2], pk[3]);
        }
    }
}

/* t0=o t1=cosb t2=sinb t3=pos · i0=n_tok i1=n_head i2=D i3=rd i4=pos0 i5=per_row (token t at pos[t]: batched decode) */
__device__ __forceinline__ void d_rope_inverse_o(__nv_bfloat16* __restrict__ o, const float* __restrict__ cosb, const float* __restrict__ sinb,
                                                 unsigned n_tok, unsigned n_head, unsigned D, unsigned rd, unsigned pos0, unsigned slice,
                                                 unsigned nblk, const int* __restrict__ pos, unsigned per_row = 0) {
    using namespace plow_cmp;
    if (!rd) return;
    if (pos != nullptr && !per_row) pos0 = (unsigned)pos[0];
    const unsigned h2 = rd / 2u, c0 = D - rd;
    const size_t n = (size_t)n_tok * n_head * h2;
    uint32_t* o32 = reinterpret_cast<uint32_t*>(o);
    for (size_t r = (size_t)slice * blockDim.x + threadIdx.x; r < n; r += (size_t)nblk * blockDim.x) {
        const size_t row = r / h2;
        const unsigned m = (unsigned)(r % h2), t = (unsigned)(row / n_head);
        const size_t p = (size_t)(per_row ? (unsigned)pos[t] : pos0 + t) * h2 + m;
        uint32_t* v = o32 + (row * D + c0) / 2u + m;
        const uint32_t u = *v;
        const float x0 = __uint_as_float(u << 16), x1 = __uint_as_float(u & 0xffff0000u);
        const float c = cosb[p], sn = -sinb[p];
        *v = (uint32_t)to_bf(x0 * c - x1 * sn) | ((uint32_t)to_bf(x0 * sn + x1 * c) << 16);
    }
}

/* PLOW_DOP_QWEN_HEADNORM_ROPE (142) in DeepSeek-V4.1's form: no norm, interleaved (rotary bit 31)
 * rope of [rot_offset, rot_offset + rotary) at pos[row / heads], f32 tables, one bf16 round; the
 * rest of the row is copied. Thread per 8 elements; needs dim, rot_offset, rotary multiples of 8. */
__device__ __forceinline__ void d_rope_interleaved_rows(__nv_bfloat16* out, const __nv_bfloat16* in,
                                                        const float* __restrict__ cosb, const float* __restrict__ sinb,
                                                        const int* __restrict__ positions, unsigned heads, unsigned dim, unsigned rotary,
                                                        unsigned rows, unsigned rot_offset, unsigned slice, unsigned nblk) {
    using namespace plow_cmp;
    if (dim % 8u || rot_offset % 8u || rotary % 8u) __trap();
    const unsigned half = rotary / 2u, d8 = dim / 8u;
    const uint4* i4 = reinterpret_cast<const uint4*>(in);
    uint4* o4 = reinterpret_cast<uint4*>(out);
    if (out == in) { /* in place: only the rotary chunks move */
        const unsigned r8 = rotary / 8u;
        const size_t n = (size_t)rows * heads * r8;
        for (size_t e0 = (size_t)slice * blockDim.x + threadIdx.x; e0 < n; e0 += (size_t)nblk * blockDim.x) {
            const size_t rh = e0 / r8, e = rh * d8 + rot_offset / 8u + e0 % r8;
            const size_t p = (size_t)positions[rh / heads] * half + (e0 % r8) * 4u;
            const float4 cv = *reinterpret_cast<const float4*>(cosb + p), sv = *reinterpret_cast<const float4*>(sinb + p);
            const float cs[4] = {cv.x, cv.y, cv.z, cv.w}, sn[4] = {sv.x, sv.y, sv.z, sv.w};
            const uint4 u = i4[e];
            uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
            for (int q = 0; q < 4; q++) {
                const float x0 = __uint_as_float(w[q] << 16), x1 = __uint_as_float(w[q] & 0xffff0000u);
                w[q] = (uint32_t)to_bf(x0 * cs[q] - x1 * sn[q]) | ((uint32_t)to_bf(x0 * sn[q] + x1 * cs[q]) << 16);
            }
            o4[e] = make_uint4(w[0], w[1], w[2], w[3]);
        }
        return;
    }
    const size_t n = (size_t)rows * heads * d8;
    for (size_t e = (size_t)slice * blockDim.x + threadIdx.x; e < n; e += (size_t)nblk * blockDim.x) {
        const unsigned c = (unsigned)(e % d8) * 8u;
        uint4 u = i4[e];
        if (c >= rot_offset && c < rot_offset + rotary) {
            const size_t p = (size_t)positions[(e / d8) / heads] * half + ((c - rot_offset) >> 1);
            const float4 cv = *reinterpret_cast<const float4*>(cosb + p), sv = *reinterpret_cast<const float4*>(sinb + p);
            const float cs[4] = {cv.x, cv.y, cv.z, cv.w}, sn[4] = {sv.x, sv.y, sv.z, sv.w};
            uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
            for (int q = 0; q < 4; q++) {
                const float x0 = __uint_as_float(w[q] << 16), x1 = __uint_as_float(w[q] & 0xffff0000u);
                w[q] = (uint32_t)to_bf(x0 * cs[q] - x1 * sn[q]) | ((uint32_t)to_bf(x0 * sn[q] + x1 * cs[q]) << 16);
            }
            u = make_uint4(w[0], w[1], w[2], w[3]);
        }
        o4[e] = u;
    }
}
