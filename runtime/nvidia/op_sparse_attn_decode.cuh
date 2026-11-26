/* op_sparse_attn_decode.cuh -- DeepSeek-V4.1 sparse attention, one decode token per slot.
 *
 * Reference: inference/kernel.py sparse_attn over kv = cat(window ring, compressed cache) with
 * topk = cat(window idxs, compressed idxs), one query row per slot (model.py Attention.forward at
 * start_pos > 0). The KV row is ONE 512-wide vector per position shared by every head (it is both
 * K and V), so the whole head group is the MMA M dimension: at TP4 a rank holds 16 heads, exactly
 * one m16 tile, and every KV row crosses HBM once.
 *
 * Virtual rows per slot b: v < nwin(b) = min(pos[b] + 1, W) -> ring[b][v] (ring order does not
 * matter to softmax); nwin <= v < nwin + topk -> cmp[b][idx[b][v - nwin]], skipped when < 0.
 *
 * Work item = (slot, 16-head group, split). A split walks a 32-row-aligned share of the virtual
 * rows, keeps the online softmax (running max starts at -1e30, as the reference's finite bound),
 * and either writes the finished rows (nsplit == 1) or an unnormalized partial that
 * d_sparse_attn_merge folds with the f32 sink after a grid barrier. The fold is grid-wide on
 * purpose: one CTA folding a slot's nsplit x 32 KB partials is the B=1 critical path.
 *
 * nsplit: the largest value with B * (H/16) * nsplit <= grid CTAs (a second round of items costs a
 * whole item's latency), at most ceil(rows / 32). Measured on H200 (H=16, 640 rows, attn + merge):
 * B=1 nsplit 20 8.4 us; B=16 nsplit 8 10.3 us (nsplit 9 = 144 items: 15.2 us); B=64 nsplit 2
 * 21.6 us = 2.0 TB/s. A 32-row tile costs ~1.3 us of serialized S -> softmax -> PV per CTA, which
 * is what bounds B=1..16 (64-row tiles without the cross-warp S reduction are the next step).
 */
#pragma once
#include <cuda_bf16.h>
#include <stdint.h>

namespace plow_sad {

constexpr unsigned D = 512, TR = 32, NS = 4, CH = D / 8; /* CH = 16 B chunks per row */
constexpr unsigned PST = TR + 8;                         /* P row stride (bf16), conflict-free ldmatrix */
constexpr unsigned Q_BYTES = 16 * D * 2, KV_BYTES = TR * D * 2;
constexpr unsigned SP_FLOATS = 2 * 16 * (TR + 1);
constexpr unsigned MAX_ROWS = 2048; /* one split's virtual rows: W + topk bound */
constexpr unsigned SMEM_BYTES = Q_BYTES + NS * KV_BYTES + SP_FLOATS * 4 + 16 * PST * 2 + 3 * 16 * 4 + MAX_ROWS * 4;
/* per (slot, group, split) partial: 16 x D f32 accumulators + 16 (m, l) pairs */
constexpr unsigned PART_FLOATS = 16 * D + 32;
__host__ __device__ constexpr size_t scratch_bytes(unsigned slots, unsigned groups, unsigned nsplit) {
    return (size_t)slots * groups * nsplit * PART_FLOATS * 4;
}

__device__ __forceinline__ unsigned swz(unsigned row, unsigned chunk) { return row * CH + (chunk ^ (row & 7u)); }

__device__ __forceinline__ void cp16(uint32_t dst, const void* src, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void ldsm4(uint32_t a, uint32_t (&r)[4]) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm4t(uint32_t a, uint32_t (&r)[4]) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void mma16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

} // namespace plow_sad

/* o [B][H][D] bf16; q [B][H][D] bf16; ring [B][W][D]; cmp [B][cmp_stride][D] (may be null when
 * topk == 0); idx [B][topk] i32; pos [B] i32; sink [H] f32; scratch = scratch_bytes(B, H/16,
 * nsplit), unused when nsplit == 1. */
__device__ void d_sparse_attn_decode(__nv_bfloat16* __restrict__ o, const __nv_bfloat16* __restrict__ q,
                                     const __nv_bfloat16* __restrict__ ring, const __nv_bfloat16* __restrict__ cmp,
                                     const int* __restrict__ idx, const int* __restrict__ pos, const float* __restrict__ sink,
                                     float* scratch, unsigned B, unsigned H, unsigned W, unsigned cmp_stride, unsigned topk,
                                     unsigned nsplit, float scale, unsigned slice, unsigned nblk, unsigned char* smem) {
    using namespace plow_sad;
    const unsigned tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const unsigned groups = H / 16;
    __nv_bfloat16* qs = (__nv_bfloat16*)smem;
    __nv_bfloat16* kvs = (__nv_bfloat16*)(smem + Q_BYTES);
    float* sp = (float*)(smem + Q_BYTES + NS * KV_BYTES);
    __nv_bfloat16* ps = (__nv_bfloat16*)(sp + SP_FLOATS);
    float* rowm = (float*)(ps + 16 * PST);
    float* rowl = rowm + 16;
    float* rowc = rowl + 16;
    int* src_row = (int*)(rowc + 16); /* this split's rows: ring row v, W + compressed row, or -1 */
    const uint32_t qs_a = (uint32_t)__cvta_generic_to_shared(qs), kv_a = (uint32_t)__cvta_generic_to_shared(kvs),
                   ps_a = (uint32_t)__cvta_generic_to_shared(ps);
    const float sl2 = scale * 1.4426950408889634f; /* exp(x*scale - m) = exp2(x*sl2 - m*log2e) */
    const unsigned n_work = B * groups * nsplit;

    for (unsigned w = slice; w < n_work; w += nblk) {
        const unsigned s = w % nsplit, g = (w / nsplit) % groups, b = w / (nsplit * groups);
        const unsigned nwin = min((unsigned)pos[b] + 1u, W), rows = nwin + topk;
        const unsigned per = ((rows + nsplit - 1) / nsplit + TR - 1) / TR * TR;
        const unsigned lo = min(s * per, rows), hi = min(lo + per, rows);
        const __nv_bfloat16* ringb = ring + (size_t)b * W * D;
        const __nv_bfloat16* cmpb = cmp ? cmp + (size_t)b * cmp_stride * D : nullptr;
        const int* idxb = idx ? idx + (size_t)b * topk : nullptr;

        __syncthreads(); /* previous item's smem readers are done */
        const __nv_bfloat16* qg = q + ((size_t)b * H + g * 16) * D;
        for (unsigned c = tid; c < 16 * CH; c += blockDim.x)
            cp16(qs_a + swz(c / CH, c % CH) * 16, qg + (size_t)c * 8, true);
        asm volatile("cp.async.commit_group;\n");
        for (unsigned v = lo + tid; v < hi; v += blockDim.x)
            src_row[v - lo] = v < nwin ? (int)v : (idxb[v - nwin] >= 0 ? (int)W + idxb[v - nwin] : -1);
        __syncthreads();
        /* stage t's rows: 32 rows x 64 chunks, 8 per thread; thread owns chunk (tid % 64) of rows tid/64 + 4k */
        auto load = [&](unsigned t0, unsigned st) {
            const unsigned ck = tid % CH;
#pragma unroll
            for (unsigned k = 0; k < TR * CH / 256; k++) {
                const unsigned r = tid / CH + k * (256 / CH), v = t0 + r;
                const int sr = v < hi ? src_row[v - lo] : -1;
                const bool ok = sr >= 0;
                const __nv_bfloat16* src = !ok ? ringb : sr < (int)W ? ringb + (size_t)sr * D : cmpb + (size_t)(sr - (int)W) * D;
                cp16(kv_a + st * KV_BYTES + swz(r, ck) * 16, src + ck * 8, ok);
            }
        };
        const unsigned ntile = (hi - lo + TR - 1) / TR;
#pragma unroll
        for (unsigned i = 0; i + 1 < NS; i++) {
            if (i < ntile) load(lo + i * TR, i);
            asm volatile("cp.async.commit_group;\n");
        }
        if (tid < 16) rowm[tid] = -1e30f, rowl[tid] = 0.f;

        float acc[8][4];
#pragma unroll
        for (int j = 0; j < 8; j++) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;

        for (unsigned t = 0; t < ntile; t++) {
            const unsigned st = t % NS, t0 = lo + t * TR;
            asm volatile("cp.async.wait_group %0;\n" ::"n"(NS - 2));
            __syncthreads(); /* tile t landed; every warp is past tile t-1, whose stage refills now */
            if (t + NS - 1 < ntile) load(t0 + (NS - 1) * TR, (t + NS - 1) % NS);
            asm volatile("cp.async.commit_group;\n");
            /* S partial: warp -> n-tile (warp & 3) of 8 rows, k half (warp >> 2) of 256 */
            {
                const unsigned nt = warp & 3u, kh = warp >> 2;
                float cs[4][4] = {};
                const uint32_t kst = kv_a + st * KV_BYTES;
#pragma unroll
                for (unsigned ks = 0; ks < 16; ks += 2) {
                    const unsigned kc = kh * 32 + ks * 2; /* chunk index of k0 */
                    uint32_t a0[4], a1[4], bb[4];
                    ldsm4(qs_a + swz(lane & 15u, kc + (lane >> 4)) * 16, a0);
                    ldsm4(qs_a + swz(lane & 15u, kc + 2 + (lane >> 4)) * 16, a1);
                    ldsm4(kst + swz(nt * 8 + (lane & 7u), kc + (lane >> 3)) * 16, bb);
                    mma16816(cs[(ks >> 1) & 1u], a0, bb[0], bb[1]);
                    mma16816(cs[2 + ((ks >> 1) & 1u)], a1, bb[2], bb[3]);
                }
                float c[4];
#pragma unroll
                for (int u = 0; u < 4; u++) c[u] = (cs[0][u] + cs[1][u]) + (cs[2][u] + cs[3][u]);
                const unsigned r0 = lane >> 2, c0 = nt * 8 + (lane & 3u) * 2;
                float* spk = sp + kh * 16 * (TR + 1);
                spk[r0 * (TR + 1) + c0] = c[0];
                spk[r0 * (TR + 1) + c0 + 1] = c[1];
                spk[(r0 + 8) * (TR + 1) + c0] = c[2];
                spk[(r0 + 8) * (TR + 1) + c0 + 1] = c[3];
            }
            __syncthreads();
            /* online softmax: 16 threads per head row, 2 columns each */
            {
                const unsigned r = tid >> 4, cc = (tid & 15u) * 2;
                float x[2];
#pragma unroll
                for (int u = 0; u < 2; u++) {
                    const unsigned col = cc + u, v = t0 + col;
                    const bool ok = v < hi && src_row[v - lo] >= 0;
                    x[u] = ok ? (sp[r * (TR + 1) + col] + sp[16 * (TR + 1) + r * (TR + 1) + col]) * sl2 : -INFINITY;
                }
                float mx = fmaxf(x[0], x[1]);
#pragma unroll
                for (int off = 8; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
                const float mold = rowm[r], mnew = fmaxf(mold, mx);
                const float p0 = exp2f(x[0] - mnew), p1 = exp2f(x[1] - mnew);
                float sum = p0 + p1;
#pragma unroll
                for (int off = 8; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);
                *reinterpret_cast<__nv_bfloat162*>(ps + r * PST + cc) = __floats2bfloat162_rn(p0, p1);
                __syncwarp();
                if ((tid & 15u) == 0) {
                    const float corr = exp2f(mold - mnew);
                    rowc[r] = corr;
                    rowl[r] = rowl[r] * corr + sum;
                    rowm[r] = mnew;
                }
            }
            __syncthreads();
            /* PV: warp owns output columns warp*64 .. +63 (8 n-tiles), k = 32 tile rows */
            {
                const float c_lo = rowc[lane >> 2], c_hi = rowc[(lane >> 2) + 8];
#pragma unroll
                for (int j = 0; j < 8; j++) acc[j][0] *= c_lo, acc[j][1] *= c_lo, acc[j][2] *= c_hi, acc[j][3] *= c_hi;
                const uint32_t kst = kv_a + st * KV_BYTES;
#pragma unroll
                for (unsigned kk = 0; kk < TR; kk += 16) {
                    uint32_t a[4];
                    ldsm4(ps_a + ((lane & 15u) * PST + kk + (lane >> 4) * 8) * 2, a);
#pragma unroll
                    for (int j = 0; j < 8; j += 2) {
                        uint32_t bb[4];
                        const unsigned row = kk + (lane & 7u) + ((lane >> 3) & 1u) * 8;
                        ldsm4t(kst + swz(row, warp * 8 + j + (lane >> 4)) * 16, bb);
                        mma16816(acc[j], a, bb[0], bb[1]);
                        mma16816(acc[j + 1], a, bb[2], bb[3]);
                    }
                }
            }
        }
        asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();

        const unsigned r0 = lane >> 2;
        if (nsplit == 1) {
            const float m0 = rowm[r0], m1 = rowm[r0 + 8];
            const float i0 = 1.f / (rowl[r0] + exp2f(sink[g * 16 + r0] * 1.4426950408889634f - m0));
            const float i1 = 1.f / (rowl[r0 + 8] + exp2f(sink[g * 16 + r0 + 8] * 1.4426950408889634f - m1));
            __nv_bfloat16* ob = o + ((size_t)b * H + g * 16) * D;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const unsigned col = warp * 64 + j * 8 + (lane & 3u) * 2;
                *reinterpret_cast<__nv_bfloat162*>(ob + r0 * D + col) = __floats2bfloat162_rn(acc[j][0] * i0, acc[j][1] * i0);
                *reinterpret_cast<__nv_bfloat162*>(ob + (r0 + 8) * D + col) = __floats2bfloat162_rn(acc[j][2] * i1, acc[j][3] * i1);
            }
            continue;
        }
        float* part = scratch + ((size_t)(b * groups + g) * nsplit + s) * PART_FLOATS;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const unsigned col = warp * 64 + j * 8 + (lane & 3u) * 2;
            *reinterpret_cast<float2*>(part + r0 * D + col) = make_float2(acc[j][0], acc[j][1]);
            *reinterpret_cast<float2*>(part + (r0 + 8) * D + col) = make_float2(acc[j][2], acc[j][3]);
        }
        if (tid < 16) part[16 * D + tid * 2] = rowm[tid], part[16 * D + tid * 2 + 1] = rowl[tid];
    }
}

/* Folds d_sparse_attn_decode's nsplit partials: one warp per (slot, head, column part). A warp item
 * is two dependent DRAM round trips (the (m, l) header, then the rows), so rows split into as many
 * parts (4, 2, 1) as still fit the grid's warps in one round. */
__device__ void d_sparse_attn_merge(__nv_bfloat16* __restrict__ o, const float* __restrict__ scratch, const float* __restrict__ sink,
                                    unsigned B, unsigned H, unsigned nsplit, unsigned slice, unsigned nblk) {
    using namespace plow_sad;
    const float l2e = 1.4426950408889634f;
    const unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    const unsigned np = B * H * 4 <= nblk * nw ? 4u : B * H * 2 <= nblk * nw ? 2u : 1u, nu = 4 / np;
    for (unsigned w = slice * nw + warp; w < B * H * np; w += nblk * nw) {
        const unsigned qc = w % np, b = w / np / H, h = w / np % H, g = h / 16, r = h % 16;
        const float* pb = scratch + (size_t)(b * (H / 16) + g) * nsplit * PART_FLOATS;
        /* lane k < nsplit holds split k's (m, l); nsplit <= 32 */
        float mk = -1e30f, lk = 0.f;
        if (lane < nsplit) mk = pb[lane * PART_FLOATS + 16 * D + r * 2], lk = pb[lane * PART_FLOATS + 16 * D + r * 2 + 1];
        float M = mk;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, off));
        const float wl = lane < nsplit ? exp2f(mk - M) : 0.f;
        float den = wl * lk;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) den += __shfl_xor_sync(0xffffffffu, den, off);
        den += exp2f(sink[h] * l2e - M);
        float4 a[4];
#pragma unroll
        for (int u = 0; u < 4; u++) a[u] = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll 4
        for (unsigned k = 0; k < nsplit; k++) {
            const float wk = __shfl_sync(0xffffffffu, wl, k);
            const float4* pk = reinterpret_cast<const float4*>(pb + k * PART_FLOATS + r * D) + qc * nu * 32 + lane;
#pragma unroll
            for (unsigned u = 0; u < 4; u++)
                if (u < nu) {
                    const float4 v = pk[u * 32];
                    a[u].x += wk * v.x, a[u].y += wk * v.y, a[u].z += wk * v.z, a[u].w += wk * v.w;
                }
        }
        const float inv = 1.f / den;
        __nv_bfloat162* ob = reinterpret_cast<__nv_bfloat162*>(o + ((size_t)b * H + h) * D) + (qc * nu * 32 + lane) * 2;
#pragma unroll
        for (unsigned u = 0; u < 4; u++)
            if (u < nu) {
                ob[u * 64] = __floats2bfloat162_rn(a[u].x * inv, a[u].y * inv);
                ob[u * 64 + 1] = __floats2bfloat162_rn(a[u].z * inv, a[u].w * inv);
            }
    }
}
