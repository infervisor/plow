/* Causal / sliding flash prefill at hd 256 for the sm_120 interpreter, FA2 form with split-D warp
 * pairs. A 64-row item is 4 row groups x 2 D halves: warp w owns query rows 16 * (w & 3) .. +15
 * and head dims 128 * (w >> 2) .. +127. Each warp keeps its half of Q as mma A fragments and its
 * half of O as f32 accumulators in registers (32 + 64 per lane; the full-D FA2 tile would need 192).
 * Per 32-key tile both warps of a pair compute the partial S over their D half, swap it through
 * shared memory (one 64-thread named barrier) and add it (lo + hi, commutative, so both hold the
 * same S), then run the same online softmax and P.V over their own D half with P in registers.
 * K/V go through a 2-stage cp.async ring, one block barrier per tile. Items run heaviest first
 * (last query tile first), heads fastest, so the heads of a KV group stream the same tiles.
 *
 * Contract of d_flash_prefill's single-request path (req == nullptr): bf16 Q/K/V, f32 scores and
 * accumulation, exp2 with the log2e-folded scale, P rounded to bf16 for P.V, l summed from the f32
 * P; window (sliding) masks per row; nsplit 1 normalises into O (bf16), nsplit > 1 writes the
 * (Opart, mlpart) partials FlashMerge folds. When the grid already has an item per block without
 * splitting (q tiles x heads >= nblk), split 0 covers the whole KV range and the other splits are
 * written empty (m = FA_NEG_INF, l = 0, O = 0): the merge result is the same and the causal items
 * no longer pin every block to one (head, split) pair. */
#pragma once
#include "op_flash_pf128.cuh" /* fp2_* mma / ldmatrix / cp.async helpers */

#define F256_HD 256
#define F256_BQ 64
#define F256_BKV 32
#define F256_LD (F256_HD + 8) /* 528 B rows: conflict-free ldmatrix phases */
#define F256_RING_BYTES (4 * F256_BKV * F256_LD * 2) /* K and V, 2 stages each */
#define F256_XCH_BYTES (8 * 4 * 32 * 16)            /* per warp: 16 x 32 partial scores */
#define F256_SMEM_BYTES (F256_RING_BYTES + F256_XCH_BYTES)

#ifndef F256_INLINE
#define F256_INLINE __noinline__
#endif

static __device__ F256_INLINE void d_flash_pf256(float* __restrict__ Opart, float* __restrict__ mlpart,
                                                 const __nv_bfloat16* __restrict__ Q,
                                                 const __nv_bfloat16* __restrict__ K,
                                                 const __nv_bfloat16* __restrict__ V,
                                                 __nv_bfloat16* __restrict__ O, unsigned seq_q,
                                                 unsigned seq_kv, unsigned n_head,
                                                 unsigned n_kv_head, unsigned q_pos0,
                                                 unsigned window, unsigned nsplit,
                                                 unsigned kv_stride, unsigned kv_mask, float scale,
                                                 unsigned slice, unsigned nblk, float* lds) {
    constexpr int HD = F256_HD, BQ = F256_BQ, BKV = F256_BKV, LD = F256_LD, HH = HD / 2;
    if (blockDim.x != 256u || n_kv_head == 0u || n_head % n_kv_head || nsplit == 0u) {
        __trap();
        return;
    }
    __nv_bfloat16* Ks = reinterpret_cast<__nv_bfloat16*>(lds); /* [2][BKV][LD] */
    __nv_bfloat16* Vs = Ks + 2 * BKV * LD;                     /* [2][BKV][LD] */
    float4* Xs = reinterpret_cast<float4*>(Vs + 2 * BKV * LD); /* [8 warps][BKV / 8][32 lanes] */
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, t4 = lane & 3, rg = warp & 3, dh = warp >> 2;
    const unsigned gqa = n_head / n_kv_head;
    const float lscale = scale * 1.4426950408889634f;

    const unsigned nqt = (seq_q + BQ - 1) / BQ;
    const bool collapse = nsplit > 1u && nqt * n_head >= nblk;
    const unsigned nsw = collapse ? 1u : nsplit;
    const unsigned n_work = nqt * n_head * nsw;

    for (unsigned w = slice; w < n_work; w += nblk) {
        const unsigned h = w % n_head, rest = w / n_head, sp = rest % nsw;
        const unsigned q0 = (nqt - 1u - rest / nsw) * BQ;
        const unsigned hkv = h / gqa;
        const __nv_bfloat16* Kb = K + (size_t)hkv * kv_stride * HD;
        const __nv_bfloat16* Vb = V + (size_t)hkv * kv_stride * HD;
        const unsigned per = (seq_kv + nsw - 1) / nsw;
        const unsigned lo = sp * per < seq_kv ? sp * per : seq_kv;
        const unsigned hi = lo + per < seq_kv ? lo + per : seq_kv;
        const unsigned qend = q0 + BQ < seq_q ? q0 + BQ : seq_q;
        const unsigned kcap = q_pos0 + qend; /* causal: no key past the tile's newest row */
        const unsigned kend = kcap < hi ? kcap : hi;
        unsigned kbeg = lo;
        if (window && q_pos0 + q0 + 1u > window) {
            const unsigned wlo = q_pos0 + q0 + 1u - window; /* oldest key the first row sees */
            if (wlo > kbeg) kbeg = wlo;
        }
        const int nt = kend > kbeg ? (int)((kend - kbeg + BKV - 1) / BKV) : 0;

        auto stage = [&](int tile, int b) {
            const unsigned kv0 = kbeg + (unsigned)tile * BKV;
            __nv_bfloat16* kd = Ks + b * BKV * LD;
            __nv_bfloat16* vd = Vs + b * BKV * LD;
#pragma unroll
            for (int i = 0; i < BKV * (HD / 8) / 256; i++) {
                const int L = tid + i * 256, row = L >> 5, c8 = (L & 31) * 8;
                const unsigned kv = kv0 + (unsigned)row;
                const bool in = kv < kend;
                const size_t off = (size_t)(kv & kv_mask) * HD + c8;
                fp2_cp16(kd + row * LD + c8, in ? Kb + off : Kb, in ? 16 : 0);
                fp2_cp16(vd + row * LD + c8, in ? Vb + off : Vb, in ? 16 : 0);
            }
            asm volatile("cp.async.commit_group;\n" ::);
        };

        __syncthreads(); /* the previous item's last tile is read out before the ring restages */
        if (nt > 0) stage(0, 0);

        const unsigned ra = q0 + (unsigned)rg * 16u + (unsigned)g, rb = ra + 8u;
        const bool va = ra < seq_q, vb = rb < seq_q;
        const __nv_bfloat16* Qa = Q + ((size_t)ra * n_head + h) * HD + dh * HH;
        const __nv_bfloat16* Qb = Q + ((size_t)rb * n_head + h) * HD + dh * HH;
        unsigned qf[HH / 16][4];
#pragma unroll
        for (int kf = 0; kf < HH / 16; kf++) {
            const int c = kf * 16 + 2 * t4;
            qf[kf][0] = va ? *reinterpret_cast<const unsigned*>(Qa + c) : 0u;
            qf[kf][1] = vb ? *reinterpret_cast<const unsigned*>(Qb + c) : 0u;
            qf[kf][2] = va ? *reinterpret_cast<const unsigned*>(Qa + c + 8) : 0u;
            qf[kf][3] = vb ? *reinterpret_cast<const unsigned*>(Qb + c + 8) : 0u;
        }
        const unsigned abs_a = q_pos0 + ra, abs_b = q_pos0 + rb;
        const unsigned warp_lo = q_pos0 + q0 + (unsigned)rg * 16u; /* oldest row of this warp */
        float o[HH / 8][4];
#pragma unroll
        for (int j = 0; j < HH / 8; j++) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
        float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
        float4* xme = Xs + warp * (BKV / 8) * 32 + lane;
        const float4* xpa = Xs + (warp ^ 4) * (BKV / 8) * 32 + lane;

        for (int t = 0; t < nt; t++) {
            const int buf = t & 1;
            asm volatile("cp.async.wait_group 0;\n" ::);
            __syncthreads(); /* tile t landed for all; tile t-1's buffer and the swap slots are free */
            if (t + 1 < nt) stage(t + 1, buf ^ 1);
            const unsigned kv0 = kbeg + (unsigned)t * BKV;
            const __nv_bfloat16* Kt = Ks + buf * BKV * LD + dh * HH;
            const __nv_bfloat16* Vt = Vs + buf * BKV * LD + dh * HH;

            float s[BKV / 8][4];
#pragma unroll
            for (int j = 0; j < BKV / 8; j++) s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
#pragma unroll
            for (int kf = 0; kf < HH / 16; kf++) {
#pragma unroll
                for (int np = 0; np < BKV / 16; np++) {
                    unsigned b[4];
                    fp2_ldsm_x4(b, Kt + (np * 16 + (lane & 7) + ((lane >> 4) << 3)) * LD + kf * 16 +
                                       ((lane >> 3) & 1) * 8);
                    fp2_mma(s[2 * np], qf[kf], b[0], b[1]);
                    fp2_mma(s[2 * np + 1], qf[kf], b[2], b[3]);
                }
            }
#pragma unroll
            for (int j = 0; j < BKV / 8; j++) xme[j * 32] = make_float4(s[j][0], s[j][1], s[j][2], s[j][3]);
            asm volatile("bar.sync %0, 64;\n" ::"r"(1 + rg) : "memory");
#pragma unroll
            for (int j = 0; j < BKV / 8; j++) {
                const float4 x = xpa[j * 32];
                s[j][0] += x.x;
                s[j][1] += x.y;
                s[j][2] += x.z;
                s[j][3] += x.w;
            }
            /* edge tile for this warp: past hi (zero-filled rows), the causal diagonal or the window */
            if (kv0 + BKV > hi || kv0 + BKV - 1u > warp_lo ||
                (window && warp_lo + 15u - kv0 >= window)) {
#pragma unroll
                for (int j = 0; j < BKV / 8; j++)
#pragma unroll
                    for (int e = 0; e < 4; e++) {
                        const unsigned kv = kv0 + (unsigned)(j * 8 + 2 * t4 + (e & 1));
                        const unsigned qa = e < 2 ? abs_a : abs_b;
                        if (kv >= hi || kv > qa || (window && qa - kv >= window)) s[j][e] = -INFINITY;
                    }
            }
            float x0 = s[0][0], x1 = s[0][2];
#pragma unroll
            for (int j = 0; j < BKV / 8; j++) {
                x0 = fmaxf(x0, fmaxf(s[j][0], s[j][1]));
                x1 = fmaxf(x1, fmaxf(s[j][2], s[j][3]));
            }
            x0 = fmaxf(x0, __shfl_xor_sync(0xffffffffu, x0, 1));
            x0 = fmaxf(x0, __shfl_xor_sync(0xffffffffu, x0, 2));
            x1 = fmaxf(x1, __shfl_xor_sync(0xffffffffu, x1, 1));
            x1 = fmaxf(x1, __shfl_xor_sync(0xffffffffu, x1, 2));
            const float n0 = fmaxf(m0, x0 * lscale), n1 = fmaxf(m1, x1 * lscale);
            /* a row with no live key yet keeps m = -inf: shift by 0 so every term is ex2(-inf) = 0 */
            const float z0 = n0 == -INFINITY ? 0.f : n0, z1 = n1 == -INFINITY ? 0.f : n1;
            const float c0 = fp2_ex2(m0 - z0), c1 = fp2_ex2(m1 - z1);
            m0 = n0;
            m1 = n1;
            float p0 = 0.f, p1 = 0.f;
#pragma unroll
            for (int j = 0; j < BKV / 8; j++) {
                s[j][0] = fp2_ex2(fmaf(s[j][0], lscale, -z0));
                s[j][1] = fp2_ex2(fmaf(s[j][1], lscale, -z0));
                s[j][2] = fp2_ex2(fmaf(s[j][2], lscale, -z1));
                s[j][3] = fp2_ex2(fmaf(s[j][3], lscale, -z1));
                p0 += s[j][0] + s[j][1];
                p1 += s[j][2] + s[j][3];
            }
            l0 = l0 * c0 + p0;
            l1 = l1 * c1 + p1;
#pragma unroll
            for (int j = 0; j < HH / 8; j++) {
                o[j][0] *= c0;
                o[j][1] *= c0;
                o[j][2] *= c1;
                o[j][3] *= c1;
            }
#pragma unroll
            for (int kk = 0; kk < BKV / 16; kk++) {
                const unsigned pa[4] = {fp2_pack(s[2 * kk][0], s[2 * kk][1]),
                                        fp2_pack(s[2 * kk][2], s[2 * kk][3]),
                                        fp2_pack(s[2 * kk + 1][0], s[2 * kk + 1][1]),
                                        fp2_pack(s[2 * kk + 1][2], s[2 * kk + 1][3])};
#pragma unroll
                for (int dp = 0; dp < HH / 16; dp++) {
                    unsigned b[4];
                    fp2_ldsm_x4_t(b, Vt + (kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * LD +
                                         dp * 16 + (lane >> 4) * 8);
                    fp2_mma(o[2 * dp], pa, b[0], b[1]);
                    fp2_mma(o[2 * dp + 1], pa, b[2], b[3]);
                }
            }
        }
        l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
        l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
        l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
        l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
        if (nsplit == 1u) {
            const float i0 = l0 > 0.f ? 1.f / l0 : 0.f, i1 = l1 > 0.f ? 1.f / l1 : 0.f;
            __nv_bfloat16* Oa = O + ((size_t)ra * n_head + h) * HD + dh * HH;
            __nv_bfloat16* Ob = O + ((size_t)rb * n_head + h) * HD + dh * HH;
#pragma unroll
            for (int j = 0; j < HH / 8; j++) {
                const int c = j * 8 + 2 * t4;
                if (va) *reinterpret_cast<unsigned*>(Oa + c) = fp2_pack(o[j][0] * i0, o[j][1] * i0);
                if (vb) *reinterpret_cast<unsigned*>(Ob + c) = fp2_pack(o[j][2] * i1, o[j][3] * i1);
            }
            continue;
        }
        /* partials: [row][head][split][hd] and [row][head][split][2] (m, l in the log2 domain) */
        const unsigned s_hi = collapse ? nsplit : sp + 1u;
        for (unsigned s2 = sp; s2 < s_hi; s2++) {
            const bool own = s2 == sp;
            float* Pa = Opart + (((size_t)ra * n_head + h) * nsplit + s2) * HD + dh * HH;
            float* Pb = Opart + (((size_t)rb * n_head + h) * nsplit + s2) * HD + dh * HH;
#pragma unroll
            for (int j = 0; j < HH / 8; j++) {
                const int c = j * 8 + 2 * t4;
                if (va) *reinterpret_cast<float2*>(Pa + c) = own ? make_float2(o[j][0], o[j][1]) : make_float2(0.f, 0.f);
                if (vb) *reinterpret_cast<float2*>(Pb + c) = own ? make_float2(o[j][2], o[j][3]) : make_float2(0.f, 0.f);
            }
            if (dh == 0 && t4 == 0) {
                if (va)
                    *reinterpret_cast<float2*>(mlpart + (((size_t)ra * n_head + h) * nsplit + s2) * 2) =
                        own ? make_float2(m0 == -INFINITY ? FA_NEG_INF : m0, l0) : make_float2(FA_NEG_INF, 0.f);
                if (vb)
                    *reinterpret_cast<float2*>(mlpart + (((size_t)rb * n_head + h) * nsplit + s2) * 2) =
                        own ? make_float2(m1 == -INFINITY ? FA_NEG_INF : m1, l1) : make_float2(FA_NEG_INF, 0.f);
            }
        }
    }
}
