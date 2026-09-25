/* dsa_prep.h — DSA indexer prefill prep fusions (PLOW_DSA_PREP).
 *
 * d_dsa_k_prep: the indexer key chain LayerNorm(k_norm, weight + bias, feat 128) -> bf16 ->
 * HeadNormRope(HD 128 interleaved, skip_norm, no gamma) -> bf16 key-cache row, as ONE pass.
 * Byte-identical to d_layernorm_bias + d_headnorm_rope(_ilp): one wave per row, lane l holds
 * elements l and l + 64 — exactly the thread -> element map of d_layernorm_bias's first two waves
 * — so each wave_sum is the same addition tree and block_sum's combine (0 + p0 + p1 + the zero
 * sums of the idle waves) is replayed term for term. The LayerNorm result is rounded to bf16
 * before the rope, as the separate pair stores and reloads it. The HeadNormRope skip_norm scale
 * (x * 1.0f * 1.0f) is the identity and is dropped. */
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
                const float r = ((lane & 1u) == 0u) ? (u[e] * c[q][e] - partner * s[q][e])
                                                    : (u[e] * c[q][e] + partner * s[q][e]);
                st_act1(&og[(size_t)(out_row0 + t) * HD + lane + e * 64], f2bf(r));
            }
        }
    }
}
