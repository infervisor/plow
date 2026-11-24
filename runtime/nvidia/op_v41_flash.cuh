/* op_v41_flash.cuh -- PLOW_DOP_FLASH_MLA_PREFILL (51) in its NoPE form (i3 bit 31), DeepSeek-V4.1's
 * sparse attention (kernel.py sparse_attn) as the two partials FlashMerge folds with the sinks:
 *
 *   window arm (t7 NONE): each query attends to the keys at positions (pos - window, pos] (all of
 *     [0, pos] when window = 0) of t4; one partial, index i7 & 0xff of i7 >> 8 (i7 = 0: 0 of 1).
 *   gathered arm (t7 = union table, IndexUnionPf with 8 queries per tile): each query attends to
 *     the cache rows its union membership bit names; the tile's union is walked in i7 >> 16
 *     ceil-equal shares written as partials (i7 & 0xff) + share.
 *
 * K = V = the 512-wide latent row (t4 == t5). A work item is 8 queries x 8 heads = 64 rows sharing
 * every key tile (MQA): 8 warps, warp (mt, nq) owns rows mt*16.. for the scores of key half nq and
 * dims nq*256.. of the output. Scores are kept in log2 units (scale * log2 e); the partial is the
 * unnormalized sum of p * v with (m, l) as d_flash_merge reads them. p is rounded to bf16 for the
 * P.V product and l sums the unrounded p, as the reference does.
 */
#pragma once
#include <cuda_bf16.h>
#include <stdint.h>

namespace plow_v41fa {
constexpr unsigned D = 512, LDQ = D + 8, BN = 32, LDP = BN + 8, QP = 8, HB = 8, M = QP * HB;
constexpr unsigned SMEM_BYTES = (M * LDQ + 2 * BN * LDQ + M * LDP) * 2 + (2 * 2 * M) * 4 + 2 * BN * 8;
__device__ __forceinline__ void mma_bf16(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void ldsm_x4_trans(uint32_t* r, const void* p) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
}
__device__ __forceinline__ void ldsm_x4(uint32_t* r, const void* p) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
}
/* 16 B global -> shared, zero-filled when !valid */
__device__ __forceinline__ void cp16(void* dst, const void* src, bool valid) {
    const uint32_t d = (uint32_t)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ float ex2(float x) {
    float r;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}
__device__ __forceinline__ uint32_t pack_bf2(float a, float b) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&v);
}
}  // namespace plow_v41fa

/* t0=Opart t1=mlpart t2=Q [n_tok][n_head][512] t4=KV rows [.][512] t6=kv_len t7=union? ·
 * i1=n_head i3=(1<<31)|window i4=n_tok i5=kv_mask i6=cap i7=(gsplit<<16)|(nsplit<<8)|sp0 · f0=scale */
__device__ __forceinline__ void d_v41_sparse_flash(float* __restrict__ Opart, float* __restrict__ mlpart, const __nv_bfloat16* __restrict__ Q,
                                                   const __nv_bfloat16* __restrict__ KV, const int* __restrict__ kv_len,
                                                   const unsigned char* __restrict__ uni, unsigned n_head, unsigned window, unsigned n_tok,
                                                   unsigned kv_mask, unsigned cap, unsigned split_word, float scale, unsigned slice,
                                                   unsigned nblk, float* __restrict__ arena, unsigned arena_floats) {
    using namespace plow_v41fa;
    if (n_head % HB || blockDim.x != 256u || arena_floats * 4u < SMEM_BYTES) __trap();
    uint16_t* Qs = reinterpret_cast<uint16_t*>(arena);            // [M][LDQ]
    uint16_t* Kb = Qs + M * LDQ;                                  // [2][BN][LDQ]
    uint16_t* Ps = Kb + 2 * BN * LDQ;                             // [M][LDP]
    float* redm = reinterpret_cast<float*>(Ps + M * LDP);         // [2][M]
    float* reds = redm + 2 * M;                                   // [2][M]
    int* kpos = reinterpret_cast<int*>(reds + 2 * M);             // [2][BN]
    unsigned* kmsk = reinterpret_cast<unsigned*>(kpos + 2 * BN);  // [2][BN]
    const uint16_t* q16 = reinterpret_cast<const uint16_t*>(Q);
    const uint16_t* kv16 = reinterpret_cast<const uint16_t*>(KV);
    const unsigned tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5, g = lane >> 2, t4 = lane & 3u;
    const unsigned mt = warp & 3u, nq = warp >> 2;
    const unsigned nsplit = split_word ? (split_word >> 8) & 0xffu : 1u, sp0 = split_word & 0xffu;
    const unsigned gsplit = uni ? ((split_word >> 16) ? (split_word >> 16) : 1u) : 1u;
    const unsigned q_pos0 = (unsigned)kv_len[0] - n_tok;
    const unsigned n_packs = (n_tok + QP - 1u) / QP, n_hb = n_head / HB;
    const unsigned hdr = (n_packs * 4u + 255u) / 256u * 256u;
    const float sl2 = scale * 1.4426950408889634f;
    /* ldmatrix lane addresses: A (16 x k16) rows lane&15, cols (lane>>4)*8; a B pair (2 x n8, k16)
     * rows (lane&7) + (lane>>4)*8, cols ((lane>>3)&1)*8 */
    const unsigned a_r = lane & 15u, a_c = (lane >> 4) * 8u, b_r = (lane & 7u) + (lane >> 4) * 8u, b_c = ((lane >> 3) & 1u) * 8u;
    for (unsigned item = slice; item < n_packs * n_hb * gsplit; item += nblk) {
        const unsigned hb = item % n_hb, share = (item / n_hb) % gsplit, pk = item / (n_hb * gsplit);
        const unsigned q0 = pk * QP;
        /* the key rows: [r_lo, r_hi) of the window range or of this share of the union */
        unsigned r_lo, r_hi;
        const int* upos = nullptr;
        const unsigned* ulo = nullptr;
        if (uni) {
            const unsigned c = reinterpret_cast<const unsigned*>(uni)[pk];
            upos = reinterpret_cast<const int*>(uni + hdr + (size_t)pk * cap * 12u);
            ulo = reinterpret_cast<const unsigned*>(upos + cap);
            r_lo = (unsigned)(((unsigned long long)c * share) / gsplit);
            r_hi = (unsigned)(((unsigned long long)c * (share + 1u)) / gsplit);
        } else {
            const unsigned p_first = q_pos0 + q0, p_last = q_pos0 + min(q0 + QP, n_tok) - 1u;
            r_lo = (window && p_first + 1u > window) ? p_first + 1u - window : 0u;
            r_hi = p_last + 1u;
        }
        const unsigned n_tiles = (r_hi - r_lo + BN - 1u) / BN;
        /* tile i's rows into buffer i & 1 (threads < BN); then its cp.async group */
        auto rows = [&](unsigned i) {
            if (tid < BN) {
                const unsigned r = r_lo + i * BN + tid;
                int pos = -1;
                unsigned m = 0u;
                if (r < r_hi) {
                    if (uni) {
                        pos = upos[r];
                        m = ulo[r] & 0xffu;
                    } else {
                        pos = (int)r;
#pragma unroll
                        for (unsigned qi = 0; qi < QP; qi++) {
                            const unsigned p = q_pos0 + q0 + qi;
                            if (q0 + qi < n_tok && r <= p && (!window || p - r < window)) m |= 1u << qi;
                        }
                    }
                }
                kpos[(i & 1u) * BN + tid] = m ? pos : -1;
                kmsk[(i & 1u) * BN + tid] = m;
            }
        };
        auto load = [&](unsigned i) {
            uint16_t* Kd = Kb + (i & 1u) * BN * LDQ;
            for (unsigned ch = tid; ch < BN * D / 8u; ch += 256u) {
                const unsigned r = ch / (D / 8u), c = (ch % (D / 8u)) * 8u;
                const int pos = kpos[(i & 1u) * BN + r];
                const size_t row = pos < 0 ? 0 : (kv_mask == 0xFFFFFFFFu ? (size_t)pos : (size_t)((unsigned)pos & kv_mask));
                cp16(Kd + r * LDQ + c, kv16 + row * D + c, pos >= 0);
            }
        };
        __syncthreads();  // the previous item is done with Qs / Kb / kpos
        for (unsigned ch = tid; ch < M * D / 8u; ch += 256u) {
            const unsigned r = ch / (D / 8u), c = (ch % (D / 8u)) * 8u, t = q0 + r / HB, h = hb * HB + r % HB;
            cp16(Qs + r * LDQ + c, q16 + ((size_t)min(t, n_tok - 1u) * n_head + h) * D + c, t < n_tok);
        }
        if (n_tiles) rows(0);
        __syncthreads();
        if (n_tiles) load(0);  // Q rides in tile 0's group
        cp_commit();
        float acc[32][4];
#pragma unroll
        for (int j = 0; j < 32; j++) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
        float mrow[2] = {-INFINITY, -INFINITY}, lrow[2] = {0.f, 0.f};
        const unsigned qbit[2] = {1u << ((mt * 16u + g) / HB), 1u << ((mt * 16u + g + 8u) / HB)};
        for (unsigned i = 0; i < n_tiles; i++) {
            if (i + 1u < n_tiles) {
                rows(i + 1u);
                __syncthreads();
                load(i + 1u);
                cp_commit();
                cp_wait<1>();
            } else {
                cp_wait<0>();
            }
            __syncthreads();
            const uint16_t* Ks = Kb + (i & 1u) * BN * LDQ;
            const unsigned* km = kmsk + (i & 1u) * BN;
            /* S[16 rows][16 keys]: rows mt*16.., keys nq*16.. */
            float s[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll 8
            for (unsigned k = 0; k < D; k += 16u) {
                uint32_t af[4], b4[4];
                ldsm_x4(af, Qs + (mt * 16u + a_r) * LDQ + k + a_c);
                ldsm_x4(b4, Ks + (nq * 16u + b_r) * LDQ + k + b_c);
                mma_bf16(s[0], af, b4);
                mma_bf16(s[1], af, b4 + 2);
            }
            float pmax[2] = {-INFINITY, -INFINITY};
#pragma unroll
            for (int j = 0; j < 2; j++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const unsigned col = nq * 16u + j * 8u + t4 * 2u + (e & 1);
                    const float v = (km[col] & qbit[e >> 1]) ? s[j][e] * sl2 : -INFINITY;
                    s[j][e] = v;
                    pmax[e >> 1] = fmaxf(pmax[e >> 1], v);
                }
#pragma unroll
            for (int h = 0; h < 2; h++) {
                pmax[h] = fmaxf(pmax[h], __shfl_xor_sync(0xffffffffu, pmax[h], 1));
                pmax[h] = fmaxf(pmax[h], __shfl_xor_sync(0xffffffffu, pmax[h], 2));
            }
            if (t4 == 0) {
                redm[nq * M + mt * 16u + g] = pmax[0];
                redm[nq * M + mt * 16u + g + 8u] = pmax[1];
            }
            __syncthreads();
            float msafe[2], alpha[2];
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const unsigned row = mt * 16u + g + h * 8u;
                const float mnew = fmaxf(mrow[h], fmaxf(redm[row], redm[M + row]));
                msafe[h] = mnew == -INFINITY ? 0.f : mnew;
                alpha[h] = ex2(mrow[h] - msafe[h]);
                mrow[h] = mnew;
            }
            float psum[2] = {0.f, 0.f};
#pragma unroll
            for (int j = 0; j < 2; j++)
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const float p0 = ex2(s[j][2 * h] - msafe[h]), p1 = ex2(s[j][2 * h + 1] - msafe[h]);
                    psum[h] += p0 + p1;
                    *reinterpret_cast<uint32_t*>(Ps + (mt * 16u + g + h * 8u) * LDP + nq * 16u + j * 8u + t4 * 2u) = pack_bf2(p0, p1);
                }
#pragma unroll
            for (int h = 0; h < 2; h++) {
                psum[h] += __shfl_xor_sync(0xffffffffu, psum[h], 1);
                psum[h] += __shfl_xor_sync(0xffffffffu, psum[h], 2);
            }
            if (t4 == 0) {
                reds[nq * M + mt * 16u + g] = psum[0];
                reds[nq * M + mt * 16u + g + 8u] = psum[1];
            }
            __syncthreads();
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const unsigned row = mt * 16u + g + h * 8u;
                lrow[h] = lrow[h] * alpha[h] + reds[row] + reds[M + row];
            }
            /* O[rows mt*16..][dims nq*256..] = O * alpha + P . V over this tile's BN keys */
#pragma unroll
            for (int j = 0; j < 32; j++) {
                acc[j][0] *= alpha[0];
                acc[j][1] *= alpha[0];
                acc[j][2] *= alpha[1];
                acc[j][3] *= alpha[1];
            }
#pragma unroll
            for (unsigned kk = 0; kk < BN; kk += 16u) {
                uint32_t af[4];
                ldsm_x4(af, Ps + (mt * 16u + a_r) * LDP + kk + a_c);
#pragma unroll
                for (int j = 0; j < 32; j += 2) {
                    uint32_t b4[4];
                    ldsm_x4_trans(b4, Ks + (kk + (lane & 15u)) * LDQ + nq * 256u + j * 8u + (lane >> 4) * 8u);
                    mma_bf16(acc[j], af, b4);
                    mma_bf16(acc[j + 1], af, b4 + 2);
                }
            }
            __syncthreads();  // buffer i & 1 and Ps are rewritten by the next tiles
        }
        if (!n_tiles) cp_wait<0>();
        /* partial (sp0 + share) of nsplit, rows (t, h) */
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const unsigned row = mt * 16u + g + h * 8u, t = q0 + row / HB, head = hb * HB + row % HB;
            if (t >= n_tok) continue;
            const size_t base = ((size_t)t * n_head + head) * nsplit + sp0 + share;
            float* o = Opart + base * D + nq * 256u + t4 * 2u;
#pragma unroll
            for (int j = 0; j < 32; j++) *reinterpret_cast<float2*>(o + j * 8) = make_float2(acc[j][2 * h], acc[j][2 * h + 1]);
            if (nq == 0 && t4 == 0)
                *reinterpret_cast<float2*>(mlpart + base * 2) = make_float2(mrow[h] == -INFINITY ? -3.0e38f : mrow[h], lrow[h]);
        }
    }
    __syncthreads();
}

/* FlashMerge for these partials: one WARP per (token, head) row so a CTA keeps 8 rows in flight
 * (the CTA-per-row merge is latency-bound here: 32k rows at 2k tokens x 16 heads). Same fold as
 * d_flash_merge<512, true>: gm = max(m_s, sink), O = sum_s O_s 2^(m_s - gm) / (sum_s l_s 2^(m_s - gm)
 * + 2^(sink - gm)), sink in natural units scaled to log2. sinks: f32 when sink_f32, else bf16. */
__device__ __forceinline__ void d_v41_flash_merge(__nv_bfloat16* __restrict__ O, const float* __restrict__ Opart,
                                                  const float* __restrict__ mlpart, const void* __restrict__ sinks, bool sink_f32,
                                                  unsigned n_rows, unsigned n_head, unsigned nsplit, unsigned slice, unsigned nblk) {
    using namespace plow_v41fa;
    const unsigned lane = threadIdx.x & 31u, warps = blockDim.x >> 5;
    for (unsigned r = slice * warps + (threadIdx.x >> 5); r < n_rows * n_head; r += nblk * warps) {
        const float2* ml = reinterpret_cast<const float2*>(mlpart) + (size_t)r * nsplit;
        float gm = -3.0e38f;
        for (unsigned s = 0; s < nsplit; s++) gm = fmaxf(gm, ml[s].x);
        float sink = -3.0e38f;
        if (sinks) {
            const unsigned h = r % n_head;
            sink = (sink_f32 ? reinterpret_cast<const float*>(sinks)[h] : __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(sinks)[h])) *
                   1.4426950408889634f;
            gm = fmaxf(gm, sink);
        }
        float gl = sinks ? ex2(sink - gm) : 0.f;
        float4 acc[4] = {};
        for (unsigned s = 0; s < nsplit; s++) {
            const float2 v = ml[s];
            const float w = ex2(v.x - gm);
            gl += v.y * w;
            const float4* op = reinterpret_cast<const float4*>(Opart + ((size_t)r * nsplit + s) * D) + lane;
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const float4 o = op[i * 32];
                acc[i].x += o.x * w;
                acc[i].y += o.y * w;
                acc[i].z += o.z * w;
                acc[i].w += o.w * w;
            }
        }
        const float inv = gl > 0.f ? 1.f / gl : 0.f;
        uint2* out = reinterpret_cast<uint2*>(O + (size_t)r * D) + lane;
#pragma unroll
        for (int i = 0; i < 4; i++) out[i * 32] = make_uint2(pack_bf2(acc[i].x * inv, acc[i].y * inv), pack_bf2(acc[i].z * inv, acc[i].w * inv));
    }
}
