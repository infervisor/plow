#pragma once
/* W8A16 small-M decode GEMV on the tensor cores (mma.sync m16n8k16, bf16 in, f32 accumulate).
 *
 * The FFMA dequant walk spends ~3.5 lane-instructions per weight byte at M=1 plus one FMA per
 * byte per row, so it is compute-bound from M=2 on (12B FP8: 8.75 ms at B=1, 12.9 at B=2). Here a
 * warp owns 16 output rows: each lane loads 16 contiguous weight bytes per row per 64-wide K chunk
 * straight into registers, converts them to bf16 exactly (e4m3 -> f16 -> bf16; 3 mantissa bits
 * fit), and one mma consumes 16 rows x 16 K x 8 tokens. K inside a chunk is PERMUTED -- lane t,
 * step s covers physical k 16t+4s..16t+4s+3 -- identically for A (weights) and B (x), so the dot
 * is unchanged while every weight load is a coalesced 16-byte vector.
 *
 * Output ownership is the packet's blocked column map (as gemv_rows_fp8). A CTA owning few rows
 * splits K across its warps and reduces the partial fragments through the arena in fixed order,
 * so results are deterministic. Numerics: products exact, f32 accumulation in a different order
 * from the FFMA walk -- equivalent, not bit-equal. */

/* 64-wide K chunks in flight per lane: the single-matrix GEMV carries half the weight registers
 * of the GLU, and a 16-token tile doubles the x registers, so each gets its own depth.
 * 12B, ms/step B=1/4/16 at a uniform depth: U=4 8.26/8.75/11.21, U=2 8.50/8.89/10.79, U=8
 * 8.57/9.43/12.13 (255 registers + spills). */
#ifndef PLOW_FP8TC_U
#define PLOW_FP8TC_U 4 /* GLU */
#endif
#ifndef PLOW_FP8TC_U1
#define PLOW_FP8TC_U1 4 /* single matrix */
#endif
#define PLOW_FP8TC_ARENA_BYTES (PLOW_NV_WARPS * 32u * 16u * 4u)

__device__ __forceinline__ unsigned fp8tc_bf16x2(unsigned short two) {
    const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)two, __NV_E4M3);
    const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h));
    const __nv_bfloat162 b = __floats2bfloat162_rn(f.x, f.y);
    return *reinterpret_cast<const unsigned*>(&b);
}

__device__ __forceinline__ void fp8tc_mma(float (&c)[4], unsigned a0, unsigned a1, unsigned a2,
                                          unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

/* One 64-wide K chunk: A rows (g, g+8) as 16 bytes each, B = up to NG groups of 8 tokens. */
template <int NG, bool Glu>
__device__ __forceinline__ void fp8tc_chunk(float (&cg)[NG][4], float (&cu)[NG][4], const uint4& wg0,
                                            const uint4& wg1, const uint4& wu0, const uint4& wu1,
                                            const uint4 (&xa)[NG], const uint4 (&xb)[NG]) {
    const unsigned* g0 = reinterpret_cast<const unsigned*>(&wg0);
    const unsigned* g1 = reinterpret_cast<const unsigned*>(&wg1);
    const unsigned* u0 = reinterpret_cast<const unsigned*>(&wu0);
    const unsigned* u1 = reinterpret_cast<const unsigned*>(&wu1);
#pragma unroll
    for (int s = 0; s < 4; s++) {
        const unsigned ag0 = fp8tc_bf16x2((unsigned short)(g0[s] & 0xffffu));
        const unsigned ag2 = fp8tc_bf16x2((unsigned short)(g0[s] >> 16));
        const unsigned ag1 = fp8tc_bf16x2((unsigned short)(g1[s] & 0xffffu));
        const unsigned ag3 = fp8tc_bf16x2((unsigned short)(g1[s] >> 16));
        unsigned au0 = 0, au1 = 0, au2 = 0, au3 = 0;
        if constexpr (Glu) {
            au0 = fp8tc_bf16x2((unsigned short)(u0[s] & 0xffffu));
            au2 = fp8tc_bf16x2((unsigned short)(u0[s] >> 16));
            au1 = fp8tc_bf16x2((unsigned short)(u1[s] & 0xffffu));
            au3 = fp8tc_bf16x2((unsigned short)(u1[s] >> 16));
        }
#pragma unroll
        for (int q = 0; q < NG; q++) {
            /* x words: 8 bf16 per uint4; step s uses bf16 4s..4s+3 of the lane's 16. */
            const unsigned* xw = reinterpret_cast<const unsigned*>(s < 2 ? &xa[q] : &xb[q]);
            const unsigned b0 = xw[(s & 1) * 2], b1 = xw[(s & 1) * 2 + 1];
            fp8tc_mma(cg[q], ag0, ag1, ag2, ag3, b0, b1);
            if constexpr (Glu) fp8tc_mma(cu[q], au0, au1, au2, au3, b0, b1);
        }
    }
}

/* C[m][n] = scale[n] * dot(x[m], W[n]) (Glu: gelu(g)*u), M <= 8*NG, K % 64 == 0. */
template <int NG, bool Glu>
static __device__ void d_gemv_fp8_tc(__nv_bfloat16* C, const __nv_bfloat16* x, const uint8_t* W,
                                     const uint8_t* Wu, const float* sg, const float* su, unsigned M,
                                     unsigned N, unsigned K, unsigned slice, unsigned nblk,
                                     float* red) {
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const unsigned g = lane >> 2, t = lane & 3u;
    const unsigned per = (N + nblk - 1u) / nblk;
    const unsigned n0 = slice * per, n1 = min(n0 + per, N);
    const unsigned ntile = n1 > n0 ? (n1 - n0 + 15u) / 16u : 0u;
    unsigned ks = 1u;
    while (ks * 2u <= nw && ntile * ks * 2u <= nw) ks *= 2u;
    const unsigned tiles_per_round = nw / ks;
    const unsigned nchunk = K / 64u;
    const unsigned my_tile = warp / ks, my_k = warp % ks;
    constexpr int U0 = Glu ? PLOW_FP8TC_U : PLOW_FP8TC_U1;
    constexpr int U = NG == 1 ? U0 : (U0 > 1 ? U0 / 2 : 1);
    for (unsigned r0 = 0; r0 < ntile; r0 += tiles_per_round) {
        const unsigned tile = r0 + my_tile;
        const bool active = my_tile < tiles_per_round && tile < ntile;
        float cg[NG][4] = {}, cu[NG][4] = {};
        if (active) {
            const unsigned ra = n0 + tile * 16u + g, rb = ra + 8u;
            const bool va = ra < n1, vb = rb < n1;
            const uint8_t* wa = W + (size_t)(va ? ra : n0) * K + t * 16u;
            const uint8_t* wb = W + (size_t)(vb ? rb : n0) * K + t * 16u;
            const uint8_t* ua = Glu ? Wu + (size_t)(va ? ra : n0) * K + t * 16u : nullptr;
            const uint8_t* ub = Glu ? Wu + (size_t)(vb ? rb : n0) * K + t * 16u : nullptr;
            const __nv_bfloat16* xr[NG];
            bool vx[NG];
#pragma unroll
            for (int q = 0; q < NG; q++) {
                const unsigned m = q * 8u + g;
                vx[q] = m < M;
                xr[q] = x + (size_t)(vx[q] ? m : 0u) * K + t * 16u;
            }
            for (unsigned c0 = my_k; c0 < nchunk; c0 += ks * U) {
                uint4 w0[U], w1[U], v0[U], v1[U];
#pragma unroll
                for (int u = 0; u < U; u++) {
                    const unsigned c = c0 + (unsigned)u * ks;
                    const bool in = c < nchunk;
                    const size_t off = (size_t)c * 64u;
                    w0[u] = (in && va) ? __ldcs((const uint4*)(wa + off)) : make_uint4(0, 0, 0, 0);
                    w1[u] = (in && vb) ? __ldcs((const uint4*)(wb + off)) : make_uint4(0, 0, 0, 0);
                    if constexpr (Glu) {
                        v0[u] = (in && va) ? __ldcs((const uint4*)(ua + off)) : make_uint4(0, 0, 0, 0);
                        v1[u] = (in && vb) ? __ldcs((const uint4*)(ub + off)) : make_uint4(0, 0, 0, 0);
                    }
                }
#pragma unroll
                for (int u = 0; u < U; u++) {
                    const unsigned c = c0 + (unsigned)u * ks;
                    if (c >= nchunk) break;
                    uint4 xa[NG], xb[NG];
#pragma unroll
                    for (int q = 0; q < NG; q++) {
                        xa[q] = vx[q] ? *(const uint4*)(xr[q] + (size_t)c * 64u) : make_uint4(0, 0, 0, 0);
                        xb[q] = vx[q] ? *(const uint4*)(xr[q] + (size_t)c * 64u + 8u) : make_uint4(0, 0, 0, 0);
                    }
                    fp8tc_chunk<NG, Glu>(cg, cu, w0[u], w1[u], v0[u], v1[u], xa, xb);
                }
            }
        }
        if (ks > 1u) {
            /* partials of warps my_k > 0 -> arena; my_k == 0 adds them in k order */
            if (active && my_k > 0u) {
#pragma unroll
                for (int q = 0; q < NG; q++)
#pragma unroll
                    for (int i = 0; i < 4; i++) {
                        red[((warp * 32u + lane) * NG + q) * 8u + i] = cg[q][i];
                        if constexpr (Glu) red[((warp * 32u + lane) * NG + q) * 8u + 4 + i] = cu[q][i];
                    }
            }
            __syncthreads();
            if (active && my_k == 0u) {
                for (unsigned j = 1; j < ks; j++) {
                    const unsigned src = warp + j;
#pragma unroll
                    for (int q = 0; q < NG; q++)
#pragma unroll
                        for (int i = 0; i < 4; i++) {
                            cg[q][i] += red[((src * 32u + lane) * NG + q) * 8u + i];
                            if constexpr (Glu) cu[q][i] += red[((src * 32u + lane) * NG + q) * 8u + 4 + i];
                        }
                }
            }
            __syncthreads();
        }
        if (active && my_k == 0u) {
#pragma unroll
            for (int q = 0; q < NG; q++)
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    const unsigned n = n0 + tile * 16u + g + (i >= 2 ? 8u : 0u);
                    const unsigned m = q * 8u + t * 2u + (i & 1u);
                    if (n < n1 && m < M) {
                        float v = cg[q][i] * sg[n];
                        if constexpr (Glu) v = act_gelu_tanh(v) * (cu[q][i] * su[n]);
                        C[(size_t)m * N + n] = __float2bfloat16(v);
                    }
                }
        }
    }
}
