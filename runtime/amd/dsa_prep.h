/* dsa_prep.h — DSA indexer prefill prep fusions (PLOW_DSA_PREP).
 *
 * d_dsa_k_prep: the indexer key chain LayerNorm(k_norm, weight + bias, feat 128) -> bf16 ->
 * HeadNormRope(HD 128 interleaved, skip_norm, no gamma) -> bf16 key-cache row, as ONE pass.
 * Byte-identical to d_layernorm_bias + d_headnorm_rope(_ilp): one wave per row, lane l holds
 * elements l and l + 64 — exactly the thread -> element map of d_layernorm_bias's first two waves
 * — so each wave_sum is the same addition tree and block_sum's combine (0 + p0 + p1 + the zero
 * sums of the idle waves) is replayed term for term. The LayerNorm result is rounded to bf16
 * before the rope, as the separate pair stores and reloads it. The HeadNormRope skip_norm scale
 * (x * 1.0f * 1.0f) is the identity and is dropped.
 *
 * d_gemm_small_dual: two GemmSmall over the same A (the indexer wk and weights_proj, N = 128 and
 * 32: 128 tiles each at T8192, half the GPU apiece) in one packet, each GEMM on its own share of
 * the workgroups in proportion to its tile count. Per-tile arithmetic is d_gemm_small's. Needs
 * op_gemm.h.
 *
 * d_dsa_rope_vec: d_headnorm_rope(_ilp)<128, interleaved> with skip_norm, no gamma and a plain
 * row-major output (the indexer q rope, in place, and the key rope) as 16-byte chunks: a lane
 * owns 8 consecutive head dims, so each rope pair (2i, 2i+1) is lane-local and the cos/sin are
 * one float4 each. Bit-identical.
 *
 * The rope is written as the explicit fma(u, c, +-(partner * s)) that hipcc contracts the shipped
 * `u * c -+ partner * s` into (gfx950 ISA: v_pk_mul then v_pk_fma). Left as the expression, the
 * lane-local form here is SLP-vectorised into two v_pk_mul + v_sub/v_add first and never
 * contracted, which differs in the last bit. */
#pragma once

#ifndef DSA_KPREP_G
#define DSA_KPREP_G 4u
#endif

__device__ void d_dsa_k_prep(bf16* __restrict__ out, const bf16* __restrict__ x,
                             const bf16* __restrict__ gamma, const bf16* __restrict__ beta,
                             const float* __restrict__ cosb, const float* __restrict__ sinb,
                             const int* __restrict__ pos, unsigned ntok, unsigned feat, float eps,
                             unsigned out_row0, unsigned slice, unsigned nblk) {
    static_assert(PLOW_WAVES >= 2, "d_layernorm_bias's feat-128 map spans two waves");
    constexpr unsigned HD = 128, H2 = HD / 2, G = DSA_KPREP_G;
    const unsigned lane = threadIdx.x & 63;
    const unsigned wave_in_blk = threadIdx.x >> 6;
    const auto* xg = as_glob(x);
    const auto* gg = as_glob(gamma);
    const auto* bg = as_glob(beta);
    const auto* cg = as_glob(cosb);
    const auto* sg = as_glob(sinb);
    const auto* pg = as_glob(pos);
    auto* og = as_glob(out);
    float g[2], bb[2];
#pragma unroll
    for (unsigned e = 0; e < 2; e++) {
        g[e] = bf2f(gg[lane + e * 64]);
        bb[e] = bf2f(bg[lane + e * 64]);
    }
    for (unsigned r0 = (slice * PLOW_WAVES + wave_in_blk) * G; r0 < ntok;
         r0 += nblk * PLOW_WAVES * G) {
        float v[G][2], c[G][2], s[G][2];
        unsigned position[G];
#pragma unroll
        for (unsigned q = 0; q < G; q++) {
            const unsigned t = r0 + q < ntok ? r0 + q : ntok - 1u;
            position[q] = pg ? (unsigned)pg[t] : out_row0 + t;
#pragma unroll
            for (unsigned e = 0; e < 2; e++) v[q][e] = bf2f(xg[(size_t)t * HD + lane + e * 64]);
        }
#pragma unroll
        for (unsigned q = 0; q < G; q++) {
            const size_t p = (size_t)position[q] * H2;
#pragma unroll
            for (unsigned e = 0; e < 2; e++) {
                c[q][e] = cg[p + ((lane + e * 64) >> 1)];
                s[q][e] = sg[p + ((lane + e * 64) >> 1)];
            }
        }
#pragma unroll
        for (unsigned q = 0; q < G; q++) {
            const unsigned t = r0 + q;
            if (t >= ntok) break;
            float ps[2], pss[2];
#pragma unroll
            for (unsigned e = 0; e < 2; e++) {
                float sum = 0.0f, sq = 0.0f;
                sum += v[q][e];
                sq += v[q][e] * v[q][e];
                ps[e] = wave_sum(sum);
                pss[e] = wave_sum(sq);
            }
            float ts = 0.0f, tss = 0.0f;
#pragma unroll
            for (int w = 0; w < PLOW_WAVES; w++) {
                ts += w < 2 ? ps[w] : 0.0f;
                tss += w < 2 ? pss[w] : 0.0f;
            }
            const float mean = ts / (float)feat;
            const float msq = rn_ss(tss) / (float)feat;
            const float inv = rsqrtf(msq - mean * mean + eps);
            float u[2];
#pragma unroll
            for (unsigned e = 0; e < 2; e++)
                u[e] = bf2f(f2bf((v[q][e] - mean) * inv * g[e] + bb[e]));
#pragma unroll
            for (unsigned e = 0; e < 2; e++) {
                const float partner = __shfl_xor(u[e], 1, PLOW_WAVE);
                const float ps = partner * s[q][e];
                const float r = __builtin_fmaf(u[e], c[q][e], (lane & 1u) == 0u ? -ps : ps);
                st_act1(&og[(size_t)(out_row0 + t) * HD + lane + e * 64], f2bf(r));
            }
        }
    }
}

__device__ void d_gemm_small_dual(bf16* C0, bf16* C1, const bf16* A, const bf16* B0, const bf16* B1,
                                  unsigned M, unsigned N0, unsigned N1, unsigned K, unsigned slice,
                                  unsigned nblk, bf16* lds) {
    const unsigned tm = (M + GM_SM_BM - 1) / GM_SM_BM;
    const unsigned t0 = tm * ((N0 + GM_SM_BN - 1) / GM_SM_BN), t1 = tm * ((N1 + GM_SM_BN - 1) / GM_SM_BN);
    if (nblk < 2u) {
        d_gemm_small(C0, A, B0, M, N0, K, slice, nblk, lds);
        __syncthreads();
        d_gemm_small(C1, A, B1, M, N1, K, slice, nblk, lds);
        return;
    }
    unsigned nb0 = (unsigned)(((unsigned long long)nblk * t0 + (t0 + t1) / 2) / (t0 + t1));
    nb0 = nb0 < 1u ? 1u : (nb0 > nblk - 1u ? nblk - 1u : nb0);
    if (slice < nb0)
        d_gemm_small(C0, A, B0, M, N0, K, slice, nb0, lds);
    else
        d_gemm_small(C1, A, B1, M, N1, K, slice - nb0, nblk - nb0, lds);
}

#ifndef DSA_ROPE_G
#define DSA_ROPE_G 4u
#endif
__device__ void d_dsa_rope_vec(bf16* out, const bf16* x, const float* __restrict__ cosb,
                               const float* __restrict__ sinb, const int* __restrict__ pos,
                               unsigned ntok, unsigned nhead, unsigned out_row0, unsigned slice,
                               unsigned nblk) {
    constexpr unsigned G = DSA_ROPE_G, CPH = 128 / 8; /* 16-byte chunks per head */
    const unsigned per_tok = nhead * CPH, total = ntok * per_tok;
    const auto* cg = as_glob(cosb);
    const auto* sg = as_glob(sinb);
    const auto* pg = as_glob(pos);
    bf16* const ob = out + (size_t)out_row0 * nhead * 128;
    for (unsigned b0 = slice * G * PLOW_THREADS; b0 < total; b0 += nblk * G * PLOW_THREADS) {
        bf16v8 v[G];
        float4 c[G], s[G];
#pragma unroll
        for (unsigned q = 0; q < G; q++) {
            const unsigned id = b0 + q * PLOW_THREADS + threadIdx.x;
            const unsigned ic = id < total ? id : total - 1u;
            const unsigned t = ic / per_tok;
            const unsigned position = pg ? (unsigned)pg[t] : out_row0 + t;
            const size_t p = (size_t)position * 64 + (ic % CPH) * 4;
            v[q] = ld_glob8(x + (size_t)ic * 8);
            c[q] = *(const PLOW_GLOB float4*)(cg + p);
            s[q] = *(const PLOW_GLOB float4*)(sg + p);
        }
#pragma unroll
        for (unsigned q = 0; q < G; q++) {
            const unsigned id = b0 + q * PLOW_THREADS + threadIdx.x;
            if (id >= total) break;
            const float cc[4] = {c[q].x, c[q].y, c[q].z, c[q].w};
            const float ss[4] = {s[q].x, s[q].y, s[q].z, s[q].w};
            bf16v8 o;
#pragma unroll
            for (unsigned k = 0; k < 8; k++) {
                const float u = bf2f(v[q][k]), partner = bf2f(v[q][k ^ 1u]);
                const float ps = partner * ss[k >> 1];
                const float r = __builtin_fmaf(u, cc[k >> 1], (k & 1u) == 0u ? -ps : ps);
                o[k] = f2bf(r);
            }
            st_glob8(ob + (size_t)id * 8, o);
        }
    }
}
