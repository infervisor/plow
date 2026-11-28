/* op_index_pf.cuh -- DSA sparse-prefill indexer on the warp32 interpreters: ops 117/118/119 with the
 * operands and semantics of runtime/amd/op_attention_common.h (d_index_score_pf, d_index_select_pf,
 * d_index_union_pf). `pool` (117 i4 / 118 i3) makes the column axis pools: kv_len stays in tokens
 * and a row's causal bound is (q_pos0 + t + 1) / pool (model.py compress_lens).
 *
 * 117: score[t][s] = scale * sum_h w[t][h] * relu(q[t][h] . k[s]), f32, written for s < row bound.
 *   Work item = (4 queries, a span of 64-key tiles); a warp owns one query x 16 heads, its q
 *   fragments live in registers for the span, K tiles stream through the arena.
 * 118: exact top-k per row (score desc, lowest position on ties), emitted in ascending position
 *   order; rows no longer than top_k emit the identity padded with -1. One CTA per row: four 8-bit
 *   radix passes over the order-preserving u32 of the f32 score, then an ordered compaction.
 * 119: per P-query tile, the union of the tile's selections as ascending positions with a u64
 *   membership mask each (bit q = local query q), in the table the gathered flash walks:
 *   u32 count[n_qt] (256 B aligned), then per tile i32 pos[cap] | u32 lo[cap] | u32 hi[cap].
 */
#pragma once
#include <cuda_bf16.h>
#include <stdint.h>

namespace plow_ix {
__device__ __forceinline__ float bf(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }
__device__ __forceinline__ void mma_bf16(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ uint32_t fkey(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
/* exclusive scan of v over the CTA; sh >= 33 u32; returns the prefix, *total the sum */
__device__ __forceinline__ unsigned cta_excl_scan(unsigned v, unsigned* sh, unsigned* total) {
    const unsigned lane = threadIdx.x & 31u, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
    unsigned x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const unsigned y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= (unsigned)o) x += y;
    }
    if (lane == 31u) sh[wid] = x;
    __syncthreads();
    unsigned base = 0, tot = 0;
    for (unsigned w = 0; w < nw; w++) {
        const unsigned c = sh[w];
        if (w < wid) base += c;
        tot += c;
    }
    __syncthreads();
    *total = tot;
    return base + x - v;
}
constexpr unsigned HI = 32, DI = 128, LD = DI + 8, SPAN_TILES = 16;
constexpr unsigned SCORE_ARENA_FLOATS = (64 * LD * 2 + 2 * 4 * 64 * 4) / 4;
}  // namespace plow_ix

/* t0=Score t1=Qidx t2=Kidx t3=W t4=kv_len · i0=n_tok i1=index_heads i2=kv_stride i3=index_head_dim
 * i4=pool · f0=scale */
__device__ __forceinline__ void d_index_score_pf(float* __restrict__ Score, const __nv_bfloat16* __restrict__ Qidx,
                                                 const __nv_bfloat16* __restrict__ Kidx, const __nv_bfloat16* __restrict__ W,
                                                 const int* __restrict__ kv_len, unsigned n_tok, unsigned heads, unsigned kv_stride,
                                                 unsigned dim, unsigned pool, float scale, unsigned slice, unsigned nblk,
                                                 float* __restrict__ arena, unsigned arena_floats) {
    using namespace plow_ix;
    if ((heads && heads != HI) || (dim && dim != DI) || blockDim.x != 256u || arena_floats < SCORE_ARENA_FLOATS) __trap();
    const uint16_t* q16 = reinterpret_cast<const uint16_t*>(Qidx);
    const uint16_t* k16 = reinterpret_cast<const uint16_t*>(Kidx);
    const uint16_t* w16 = reinterpret_cast<const uint16_t*>(W);
    uint16_t* Ks = reinterpret_cast<uint16_t*>(arena);                      // [64][LD]
    float(*part)[4][64] = reinterpret_cast<float(*)[4][64]>(Ks + 64 * LD);  // [2][4][64]
    const unsigned tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5, g = lane >> 2, t4 = lane & 3u;
    const unsigned qi_w = warp >> 1, hq = (warp & 1u) * 16u;
    const unsigned len_tok = (unsigned)kv_len[0], q_pos0 = len_tok - n_tok;
    const unsigned cols = len_tok / pool;
    const unsigned n_packs = (n_tok + 3u) / 4u;
    const unsigned n_spans = ((cols + 63u) / 64u + SPAN_TILES - 1u) / SPAN_TILES;
    for (unsigned item = slice; item < n_packs * n_spans; item += nblk) {
        const unsigned p = item / n_spans, sp = item % n_spans, t0 = p * 4u;
        const unsigned t_last = min(t0 + 3u, n_tok - 1u);
        const unsigned pack_end = (q_pos0 + t_last + 1u) / pool;
        const unsigned s_begin = sp * SPAN_TILES * 64u;
        if (s_begin >= pack_end) continue;  // uniform across the CTA
        const unsigned s_stop = min(s_begin + SPAN_TILES * 64u, pack_end);
        const unsigned t = t0 + qi_w;
        const bool live = t < n_tok;
        uint32_t af[8][4];
        float wh0 = 0.f, wh1 = 0.f;
        if (live) {
            const uint16_t* qa = q16 + ((size_t)t * HI + hq + g) * DI + t4 * 2u;
#pragma unroll
            for (int ks = 0; ks < 8; ks++) {
                af[ks][0] = *reinterpret_cast<const uint32_t*>(qa + ks * 16);
                af[ks][1] = *reinterpret_cast<const uint32_t*>(qa + 8 * DI + ks * 16);
                af[ks][2] = *reinterpret_cast<const uint32_t*>(qa + ks * 16 + 8);
                af[ks][3] = *reinterpret_cast<const uint32_t*>(qa + 8 * DI + ks * 16 + 8);
            }
            wh0 = bf(w16[(size_t)t * HI + hq + g]);
            wh1 = bf(w16[(size_t)t * HI + hq + g + 8u]);
        } else {
#pragma unroll
            for (int ks = 0; ks < 8; ks++) af[ks][0] = af[ks][1] = af[ks][2] = af[ks][3] = 0u;
        }
        for (unsigned s0 = s_begin; s0 < s_stop; s0 += 64u) {
            __syncthreads();
            for (unsigned ch = tid; ch < 64u * DI / 8u; ch += 256u) {
                const unsigned r = ch / (DI / 8u), c = (ch % (DI / 8u)) * 8u;
                *reinterpret_cast<uint4*>(Ks + r * LD + c) =
                    s0 + r < cols ? *reinterpret_cast<const uint4*>(k16 + (size_t)(s0 + r) * DI + c) : make_uint4(0, 0, 0, 0);
            }
            __syncthreads();
            float s[8][4];
#pragma unroll
            for (int j = 0; j < 8; j++) s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
#pragma unroll
            for (int ks = 0; ks < 8; ks++)
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    const uint16_t* kb = Ks + (j * 8 + g) * LD + ks * 16 + t4 * 2u;
                    const uint32_t bfr[2] = {*reinterpret_cast<const uint32_t*>(kb), *reinterpret_cast<const uint32_t*>(kb + 8)};
                    mma_bf16(s[j], af[ks], bfr);
                }
#pragma unroll
            for (int j = 0; j < 8; j++)
#pragma unroll
                for (int e = 0; e < 2; e++) {
                    float v = fmaf(wh0, fmaxf(s[j][e], 0.f), wh1 * fmaxf(s[j][2 + e], 0.f));
                    v += __shfl_xor_sync(0xffffffffu, v, 4);
                    v += __shfl_xor_sync(0xffffffffu, v, 8);
                    v += __shfl_xor_sync(0xffffffffu, v, 16);
                    if (g == 0) part[warp & 1u][qi_w][j * 8 + t4 * 2u + e] = v;
                }
            __syncthreads();
            {
                const unsigned qi = tid >> 6, c = tid & 63u, tt = t0 + qi, pos = s0 + c;
                if (tt < n_tok && pos < (q_pos0 + tt + 1u) / pool)
                    Score[(size_t)tt * kv_stride + pos] = (part[0][qi][c] + part[1][qi][c]) * scale;
            }
        }
    }
    __syncthreads();
}

/* t0=idx t1=Score t2=kv_len · i0=n_tok i1=top_k i2=kv_stride i3=pool. arena: 256 + 64 u32. */
__device__ __forceinline__ void d_index_select_pf(int* __restrict__ idx, const float* __restrict__ Score, const int* __restrict__ kv_len,
                                                  unsigned n_tok, unsigned top_k, unsigned kv_stride, unsigned pool, unsigned slice,
                                                  unsigned nblk, unsigned* __restrict__ arena) {
    using namespace plow_ix;
    unsigned* hist = arena;       // [256]
    unsigned* sh = arena + 256;   // [32] scan scratch
    unsigned* ctl = arena + 288;  // [4]
    const unsigned tid = threadIdx.x;
    const unsigned q_pos0 = (unsigned)kv_len[0] - n_tok;
    for (unsigned t = slice; t < n_tok; t += nblk) {
        const unsigned n = (q_pos0 + t + 1u) / pool;
        int* row = idx + (size_t)t * top_k;
        if (n <= top_k) {
            for (unsigned j = tid; j < top_k; j += blockDim.x) row[j] = j < n ? (int)j : -1;
            continue;
        }
        const float* sr = Score + (size_t)t * kv_stride;
        uint32_t prefix = 0u, himask = 0u;
        unsigned need = top_k;
        for (int pass = 0; pass < 4; pass++) {
            const unsigned sh_bits = 24u - 8u * pass;
            for (unsigned i = tid; i < 256u; i += blockDim.x) hist[i] = 0u;
            __syncthreads();
            for (unsigned s = tid; s < n; s += blockDim.x) {
                const uint32_t k = fkey(sr[s]);
                if ((k & himask) == prefix) atomicAdd(&hist[(k >> sh_bits) & 255u], 1u);
            }
            __syncthreads();
            if (tid < 32u) {  // warp 0: find the bin holding the need-th largest
                unsigned c[8], sum = 0;
#pragma unroll
                for (int j = 0; j < 8; j++) sum += (c[j] = hist[255u - tid * 8u - j]);
                unsigned incl = sum;
#pragma unroll
                for (int o = 1; o < 32; o <<= 1) {
                    const unsigned y = __shfl_up_sync(0xffffffffu, incl, o);
                    if (tid >= (unsigned)o) incl += y;
                }
                unsigned acc = incl - sum;
                if (acc < need && incl >= need) {
                    for (int j = 0; j < 8; j++) {
                        if (acc + c[j] >= need) {
                            ctl[0] = 255u - tid * 8u - j;
                            ctl[1] = acc;
                            break;
                        }
                        acc += c[j];
                    }
                }
            }
            __syncthreads();
            prefix |= ctl[0] << sh_bits;
            himask |= 255u << sh_bits;
            need -= ctl[1];
            __syncthreads();
        }
        /* keys > prefix all selected; `need` of the keys == prefix, lowest positions first */
        unsigned written = 0, eq_seen = 0;
        for (unsigned base = 0; base < n; base += blockDim.x) {
            const unsigned s = base + tid;
            unsigned sel = 0, eq = 0;
            if (s < n) {
                const uint32_t k = fkey(sr[s]);
                sel = k > prefix;
                eq = k == prefix;
            }
            unsigned eq_tot, sel_tot;
            const unsigned eq_rank = cta_excl_scan(eq, sh, &eq_tot) + eq_seen;
            if (eq && eq_rank < need) sel = 1u;
            const unsigned o = cta_excl_scan(sel, sh, &sel_tot) + written;
            if (sel && o < top_k) row[o] = (int)s;
            written += sel_tot;
            eq_seen += eq_tot;
        }
        __syncthreads();
    }
}

/* t0=union t1=umask(u64 [nblk][kv_stride]) t2=idx t3=kv_len · i0=n_tok i1=top_k i2=kv_stride i3=cap
 * i4=queries per tile (0 = 64) i5=zero_ctr. arena: 32 u32. */
__device__ __forceinline__ void d_index_union_pf(unsigned char* __restrict__ uni, unsigned long long* __restrict__ umask,
                                                 const int* __restrict__ idx, const int* __restrict__ kv_len, unsigned n_tok,
                                                 unsigned top_k, unsigned kv_stride, unsigned cap, unsigned tile_p, unsigned zero_ctr,
                                                 unsigned slice, unsigned nblk, unsigned* __restrict__ arena) {
    using namespace plow_ix;
    const unsigned tid = threadIdx.x;
    const unsigned q_pos0 = (unsigned)kv_len[0] - n_tok;
    const unsigned P = tile_p ? tile_p : 64u;
    const unsigned n_qt = (n_tok + P - 1u) / P;
    const unsigned hdr = (n_qt * 4u + 255u) / 256u * 256u;
    unsigned* cnt = reinterpret_cast<unsigned*>(uni);
    if (zero_ctr && slice == 0u && tid < 2u) reinterpret_cast<unsigned*>(uni + hdr + (size_t)n_qt * cap * 12u)[tid] = 0u;
    unsigned long long* mrow = umask + (size_t)slice * kv_stride;
    for (unsigned qt = slice; qt < n_qt; qt += nblk) {
        const unsigned q_hi = min(qt * P + P - 1u, n_tok - 1u);
        const unsigned end = min(q_pos0 + q_hi + 1u, kv_stride);
        int* upos = reinterpret_cast<int*>(uni + hdr + (size_t)qt * cap * 12u);
        unsigned* ulo = reinterpret_cast<unsigned*>(upos + cap);
        unsigned* uhi = ulo + cap;
        for (unsigned s = tid; s < end; s += blockDim.x) mrow[s] = 0ull;
        __syncthreads();
        for (unsigned e = tid; e < P * top_k; e += blockDim.x) {
            const unsigned ql = e / top_k, qi = qt * P + ql;
            if (qi >= n_tok) continue;
            const int s = idx[(size_t)qi * top_k + e % top_k];
            if (s >= 0 && (unsigned)s < end) atomicOr(&mrow[s], 1ull << ql);
        }
        __syncthreads();
        unsigned base = 0;
        for (unsigned c0 = 0; c0 < end; c0 += blockDim.x) {
            const unsigned s = c0 + tid;
            const unsigned long long m = s < end ? __ldcg(&mrow[s]) : 0ull;
            unsigned total;
            const unsigned rank = cta_excl_scan(m != 0ull, arena, &total);
            if (m && base + rank < cap) {
                upos[base + rank] = (int)s;
                ulo[base + rank] = (unsigned)m;
                uhi[base + rank] = (unsigned)(m >> 32);
            }
            base += total;
        }
        if (tid == 0) cnt[qt] = min(base, cap);
        __syncthreads();
    }
}
