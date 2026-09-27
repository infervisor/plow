/* op_gemv_k8_sm90.cuh — tensor-core K-split GEMV for 2..8 activation rows (PLOW_NV_GEMV_K8).
 *
 * The row-block walk (op_gemv_mma.cuh) gives a warp 8 WEIGHT rows as the mma N and up to 16
 * activation rows as M, so a 2..8-row rung pads M and a block's 24..63 rows keep 3..5 of its 8
 * warps busy. Here the roles swap, as in the flash score (fa_decode_qk_mma): the mma's 16 A rows
 * are 16 weight rows, its 8 B columns the activation rows, and a block's rows are cut into
 * 16-row blocks whose K range is split over the remaining warps (S = 8 / row blocks), which
 * reduce through the arena. Fragments come straight from global memory in a virtual K order
 * applied to both operands (exact dot; f32 sum order differs from the other walks).
 * H100 microbench, 8 x Veena-layer weights, grid barrier per op, M=8 (M=2 within 2%):
 *   qkv 5120x3072 13.5 us, o 3072x3072 8.9, gate|up 2x8192x3072 37.1, down 3072x8192 20.1
 * against the M=1 register walk's 13.0 / 8.7 / 36.3 / 18.8 — the 8-row rung streams at the
 * one-row rate. In the Veena B=8 step the walk measured 17.6 / 10.6 / 41.2 / 28.6.
 * CONTRACT: K % (32 * S) == 0 (callers check k8_split); M in [1, 8 * MT]; the arena holds
 * NW * 8 * 128 * MT floats. */
#pragma once

#ifndef PLOW_NV_GEMV_K8
#define PLOW_NV_GEMV_K8 0
#endif
/* k32 steps of weight loads in flight per lane (NW matrices each). */
#ifndef PLOW_NV_GEMV_K8_UNB
#define PLOW_NV_GEMV_K8_UNB 12
#endif

#if PLOW_NV_GEMV_K8
/* Up to three weight matrices stacked on the N axis (q|k|v); a row resolves its own matrix. */
struct K8Mats {
    const __nv_bfloat16* w[3];
    __nv_bfloat16* c[3];
    unsigned n[3];
};

/* Selects, not m.w[i]: a runtime index would move the struct to the local-memory stack. */
__device__ __forceinline__ void k8_row(const K8Mats& m, unsigned r, const __nv_bfloat16*& w,
                                       __nv_bfloat16*& c, unsigned& local, unsigned& ld) {
    const bool q = r < m.n[0], k = !q && r < m.n[0] + m.n[1];
    w = q ? m.w[0] : (k ? m.w[1] : m.w[2]);
    c = q ? m.c[0] : (k ? m.c[1] : m.c[2]);
    local = q ? r : (k ? r - m.n[0] : r - m.n[0] - m.n[1]);
    ld = q ? m.n[0] : (k ? m.n[1] : m.n[2]);
}

/* Warps per 16-row block: 8 / blocks, halved until it divides the k32 steps. 0 = not eligible. */
__device__ __forceinline__ unsigned k8_split(unsigned N, unsigned K, unsigned nblk) {
    const unsigned per = (N + nblk - 1u) / nblk, nrb = (per + 15u) / 16u;
    if (nrb == 0u || nrb > PLOW_NV_WARPS || (K & 31u)) return 0u;
    unsigned s = 1u;
    while (s * 2u * nrb <= PLOW_NV_WARPS) s *= 2u;
    while (s > 1u && ((K >> 5) % s)) s >>= 1;
    return s;
}

/* NW = 1: C = x W^T over the stacked matrices. NW = 2: gate|up with the GLU epilogue (m.w[0] gate,
 * m.w[1] up, one output m.c[0], N = m.n[0]). MT 8-row activation tiles: M <= 8 * MT. */
#ifndef PLOW_NV_GEMV_K8_INLINE
#define PLOW_NV_GEMV_K8_INLINE 1
#endif
#if PLOW_NV_GEMV_K8_INLINE
#define PLOW_NV_K8_ATTR __forceinline__
#else
#define PLOW_NV_K8_ATTR __noinline__
#endif
template <int NW, int MT>
__device__ PLOW_NV_K8_ATTR void d_gemv_k8(const K8Mats& m, const __nv_bfloat16* __restrict__ x,
                                          unsigned M, unsigned K, unsigned act, unsigned slice,
                                          unsigned nblk, float* red) {
    constexpr unsigned UNB = PLOW_NV_GEMV_K8_UNB / (NW * MT) > 2u ? PLOW_NV_GEMV_K8_UNB / (NW * MT) : 2u;
    constexpr unsigned CE = 128u * MT; /* reduction floats per (row block, split) */
    const unsigned N = NW == 1 ? m.n[0] + m.n[1] + m.n[2] : m.n[0];
    const unsigned S = k8_split(N, K, nblk);
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3u;
    const unsigned per = (N + nblk - 1u) / nblk;
    const unsigned n0 = slice * per, n1 = min(n0 + per, N);
    const unsigned nr = n1 > n0 ? n1 - n0 : 0u, nrb = (nr + 15u) / 16u;
    const unsigned rb = warp / S, ks = warp % S, KS = K / S, steps = KS >> 5;
    float acc[NW][MT][4];
#pragma unroll
    for (int i = 0; i < NW; i++)
#pragma unroll
        for (int j = 0; j < MT; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.0f;
    if (rb < nrb) {
        const unsigned k0 = ks * KS + t * 8u;
        const uint4* w0[NW];
        const uint4* w1[NW];
        {
            const unsigned ra = min(n0 + rb * 16u + g, n1 - 1u), rc = min(n0 + rb * 16u + g + 8u, n1 - 1u);
            const __nv_bfloat16 *wa, *wc;
            __nv_bfloat16* c_;
            unsigned la, lc, ld_;
            if constexpr (NW == 1) {
                k8_row(m, ra, wa, c_, la, ld_);
                k8_row(m, rc, wc, c_, lc, ld_);
                w0[0] = (const uint4*)(wa + (size_t)la * K + k0);
                w1[0] = (const uint4*)(wc + (size_t)lc * K + k0);
            } else {
#pragma unroll
                for (int i = 0; i < NW; i++) {
                    w0[i] = (const uint4*)(m.w[i] + (size_t)ra * K + k0);
                    w1[i] = (const uint4*)(m.w[i] + (size_t)rc * K + k0);
                }
            }
        }
        const uint4* xp[MT];
#pragma unroll
        for (int j = 0; j < MT; j++) xp[j] = (const uint4*)(x + (size_t)min(8u * j + g, M - 1u) * K + k0);
        /* Full UNB batches off advancing bases (constant load offsets), then single steps. */
        auto step = [&](const uint4 (&a0)[NW], const uint4 (&a1)[NW], const uint4 (&b)[MT]) {
#pragma unroll
            for (int i = 0; i < NW; i++)
#pragma unroll
                for (int j = 0; j < MT; j++) {
                    gvmma_mma16816(acc[i][j], a0[i].x, a1[i].x, a0[i].y, a1[i].y, b[j].x, b[j].y);
                    gvmma_mma16816(acc[i][j], a0[i].z, a1[i].z, a0[i].w, a1[i].w, b[j].z, b[j].w);
                }
        };
        unsigned s = 0;
#pragma unroll 1
        for (; s + UNB <= steps; s += UNB) {
            uint4 a0[UNB][NW], a1[UNB][NW], b[UNB][MT];
#pragma unroll
            for (unsigned u = 0; u < UNB; u++) {
#pragma unroll
                for (int i = 0; i < NW; i++) {
                    a0[u][i] = __ldcs(w0[i] + u * 4u);
                    a1[u][i] = __ldcs(w1[i] + u * 4u);
                }
#pragma unroll
                for (int j = 0; j < MT; j++) b[u][j] = __ldg(xp[j] + u * 4u);
            }
#pragma unroll
            for (unsigned u = 0; u < UNB; u++) step(a0[u], a1[u], b[u]);
#pragma unroll
            for (int i = 0; i < NW; i++) {
                w0[i] += UNB * 4u;
                w1[i] += UNB * 4u;
            }
#pragma unroll
            for (int j = 0; j < MT; j++) xp[j] += UNB * 4u;
        }
#pragma unroll 1
        for (; s < steps; s++) {
            uint4 a0[NW], a1[NW];
#pragma unroll
            for (int i = 0; i < NW; i++) {
                a0[i] = __ldcs(w0[i]);
                a1[i] = __ldcs(w1[i]);
                w0[i] += 4u;
                w1[i] += 4u;
            }
            uint4 b[MT];
#pragma unroll
            for (int j = 0; j < MT; j++) {
                b[j] = __ldg(xp[j]);
                xp[j] += 4u;
            }
            step(a0, a1, b);
        }
        /* C fragment: lane (g, t) holds rows g, g+8 of the block x activation rows 8j+2t, +1. */
#pragma unroll
        for (int i = 0; i < NW; i++) {
            float* rr = red + ((i * nrb + rb) * S + ks) * CE;
#pragma unroll
            for (int j = 0; j < MT; j++) {
                const unsigned c0 = 8u * j + 2u * t;
                rr[g * 8u * MT + c0] = acc[i][j][0];
                rr[g * 8u * MT + c0 + 1u] = acc[i][j][1];
                rr[(g + 8u) * 8u * MT + c0] = acc[i][j][2];
                rr[(g + 8u) * 8u * MT + c0 + 1u] = acc[i][j][3];
            }
        }
    }
    __syncthreads();
    for (unsigned e = threadIdx.x; e < nrb * CE; e += PLOW_NV_THREADS) {
        const unsigned b_ = e / CE, el = e % CE, row = el / (8u * MT), mm = el % (8u * MT);
        const unsigned n = n0 + b_ * 16u + row;
        if (n >= n1 || mm >= M) continue;
        float v[NW];
#pragma unroll
        for (int i = 0; i < NW; i++) {
            v[i] = 0.0f;
            for (unsigned q = 0; q < S; q++) v[i] += red[((i * nrb + b_) * S + q) * CE + el];
        }
        if constexpr (NW == 1) {
            const __nv_bfloat16* w_;
            __nv_bfloat16* c_;
            unsigned l_, ld_;
            k8_row(m, n, w_, c_, l_, ld_);
            c_[(size_t)mm * ld_ + l_] = __float2bfloat16(v[0]);
        } else {
            m.c[0][(size_t)mm * N + n] = gemma_glu_epilogue(v[0], v[1], act);
        }
    }
}
#endif
