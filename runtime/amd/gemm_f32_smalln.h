// Small-N bf16 GEMM with FP32 output (the GLM router: N = n_experts = 256, K = hidden).
//
// d_gemm_f32 runs the 256x256 prefill tile, so the router at M=1024 has 4 tiles for 256
// workgroups and walks K=6144 serially: 0.24 ms/layer on MI350X for a 3 GFLOP, 3 MB problem.
// Here one WAVE owns 16 rows x 64 columns and walks all of K with 16x16x32 bf16 MFMAs fed
// straight from global (row-major A[M][K], B[N][K], K contiguous for both operands), so M=1024
// is 256 wave units. Deterministic; accumulation order differs from the tiled body (f32
// reassociation only).
#pragma once

__device__ void d_gemm_f32_smalln(float* __restrict__ C, const bf16* __restrict__ A,
                                  const bf16* __restrict__ B, unsigned M, unsigned N, unsigned K,
                                  unsigned slice, unsigned nblk) {
    constexpr unsigned NBLK = 4, DEPTH = 4;  // 4 x 16 columns per wave; K steps of loads in flight
    const unsigned lane = threadIdx.x & 63u, wave = threadIdx.x >> 6;
    const unsigned rt = (M + 15u) / 16u, ct = (N + 16u * NBLK - 1u) / (16u * NBLK);
    const unsigned units = rt * ct, wstride = nblk * PLOW_WAVES;
    const unsigned r16 = lane & 15u, kq = (lane >> 4) * 8u;
    for (unsigned u = slice * PLOW_WAVES + wave; u < units; u += wstride) {
        const unsigned m0 = (u / ct) * 16u, n0 = (u % ct) * 16u * NBLK;
        const unsigned am = m0 + r16;
        const bf16* ap = A + (size_t)(am < M ? am : M - 1u) * K + kq;
        const bf16* bp[NBLK];
#pragma unroll
        for (unsigned j = 0; j < NBLK; j++) {
            const unsigned bn = n0 + j * 16u + r16;
            bp[j] = B + (size_t)(bn < N ? bn : N - 1u) * K + kq;
        }
        f32x4 acc[NBLK];
#pragma unroll
        for (unsigned j = 0; j < NBLK; j++) acc[j] = (f32x4)(0.0f);
        bf16x8 a[DEPTH], b[DEPTH][NBLK];
        auto load = [&](unsigned d, unsigned k) {
            __builtin_memcpy(&a[d], ap + k, 16);
#pragma unroll
            for (unsigned j = 0; j < NBLK; j++) __builtin_memcpy(&b[d][j], bp[j] + k, 16);
        };
#pragma unroll
        for (unsigned d = 0; d < DEPTH; d++) load(d, d * 32u);
        for (unsigned k = 0; k < K; k += DEPTH * 32u) {
#pragma unroll
            for (unsigned d = 0; d < DEPTH; d++) {
#pragma unroll
                for (unsigned j = 0; j < NBLK; j++) acc[j] = plow_mfma_bf16_16x16(a[d], b[d][j], acc[j]);
                const unsigned kn = k + (d + DEPTH) * 32u;
                if (kn < K) load(d, kn);
            }
        }
#pragma unroll
        for (unsigned j = 0; j < NBLK; j++) {
            const unsigned n = n0 + j * 16u + r16;
#pragma unroll
            for (unsigned i = 0; i < 4; i++) {
                const unsigned m = m0 + (lane >> 4) * 4u + i;
                if (m < M && n < N) C[(size_t)m * N + n] = acc[j][i];
            }
        }
    }
}
