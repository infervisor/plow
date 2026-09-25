// A4W4 grouped MoE GATE/UP + SiLU·up + fp4 bridge (op 85's contract), K split across waves, for
// decode-sized rows.
//
// op85 walks all of K=H behind one 64-row tile per workgroup (0.09-0.17 ms at 1-128 routed rows on
// MI350X, ~7% of the weight-stream roof); a wave that walks all of K alone is no better (latency).
// Here a workgroup unit is (one expert's 16-row subtile, 32 gate + 32 up columns) and its waves
// split K: each quantizes its own A groups straight into MFMA operands with op85's quantizer
// (one 32-element E8M0 group per lane per K-step), streams its K-slice of the weights, and the
// partial accumulators are summed in LDS in fixed wave order before op85's GLU + bridge.
// fp4 x fp4 products with power-of-two scales sum exactly in f32, so the bytes match op85.
#pragma once
#include "moe_down_a4w4_sweep.h"  // moe_sw_* MFMA helpers

template <unsigned DEPTH>
__device__ void d_moe_glu_a4w4_kw(unsigned char* __restrict__ fu, unsigned char* __restrict__ fu_scale,
                                  const bf16* __restrict__ x, const unsigned long long* __restrict__ wtab,
                                  const unsigned long long* __restrict__ stab, const int* __restrict__ meta,
                                  const unsigned* __restrict__ row_token,
                                  const unsigned* __restrict__ row_partidx, unsigned I, unsigned H,
                                  unsigned n_exp, unsigned act, float beta, float lbeta,
                                  unsigned slice, unsigned nblk, float* lds) {
    constexpr unsigned TM = 16;
    const unsigned tid = threadIdx.x, lane = tid & 63u, wave = tid >> 6, waves = blockDim.x >> 6;
    const unsigned lane_k = lane >> 4, lane_n = lane & 15u;
    const unsigned RB = H / 2, GS = H / 32, KS = H / 128, KW = (KS + waves - 1u) / waves;
    const unsigned chunks = I / 32u;
    float* red = lds;                          // [waves][4][64][4] partials
    float* br = lds + waves * 4u * 64u * 4u;   // [TM][32] GLU strip
    // The routing header (row offsets, counts, tile prefix) staged once: every unit binary-
    // searches it, and in global memory that was ~8 dependent round trips per unit.
    // Only OCCUPIED 16-row subtiles are units: occ[e] = prefix of ceil(rowcnt/16). Enumerating
    // every subtile of every 64-row tile left 3/4 of them empty at decode, and the round-robin
    // then landed all live units on 1/4 of the workgroups (8 serial units each at M8).
    int* mc = (int*)(br + TM * 32u);
    int* occ = mc + 2 * n_exp;  // [n_exp + 1]
    for (unsigned i = tid; i < 2u * n_exp; i += blockDim.x) mc[i] = meta[i];
    __syncthreads();
    if (tid == 0) {
        int acc = 0;
        for (unsigned e2 = 0; e2 < n_exp; e2++) {
            occ[e2] = acc;
            acc += (mc[n_exp + e2] + (int)TM - 1) / (int)TM;
        }
        occ[n_exp] = acc;
    }
    __syncthreads();
    const int* rowoff = mc;
    const unsigned units = (unsigned)occ[n_exp] * chunks;

    for (unsigned u = slice; u < units; u += nblk) {
        const unsigned c = u % chunks, os = u / chunks;
        unsigned lo = 0, hi = n_exp;
        while (lo + 1u < hi) {
            const unsigned mid = (lo + hi) >> 1;
            if ((unsigned)occ[mid] <= os) lo = mid; else hi = mid;
        }
        const unsigned e = lo;
        const unsigned rowbase = (unsigned)rowoff[e] + (os - (unsigned)occ[e]) * TM;
        if (wtab[(size_t)e * 3] == 0ull) continue;
        const unsigned n0 = c * 32u;
        const unsigned char* W[2] = {(const unsigned char*)(size_t)wtab[(size_t)e * 3],
                                     (const unsigned char*)(size_t)wtab[(size_t)e * 3 + 1]};
        const unsigned char* S[2] = {(const unsigned char*)(size_t)stab[(size_t)e * 3],
                                     (const unsigned char*)(size_t)stab[(size_t)e * 3 + 1]};
        const unsigned src = row_token[rowbase + lane_n];
        const bf16* xr = src == PLOW_EXPERT_UNUSED ? nullptr : x + (size_t)src * H;
        const unsigned k0 = wave * KW, k1 = min(KS, k0 + KW);

        moe_sw_f32x4 acc[4] = {};  // gate 0-15, gate 16-31, up 0-15, up 16-31
        moe_sw_i32x8 b[DEPTH][4];
        int sb[DEPTH][4];
        mpf4_a32 araw[DEPTH];
        auto loadb = [&](unsigned d, unsigned ks) {
            if (ks >= k1) return;
            // A rides the same stage as B: its raw bf16 group is in flight across the MFMAs.
            if (xr) araw[d] = mpf4_quant_load(xr + ks * 128u + lane_k * 32u);
#pragma unroll
            for (unsigned q = 0; q < 4; q++) {
                const unsigned n = n0 + (q & 1u) * 16u + lane_n;
                b[d][q] = moe_sw_fp4(W[q >> 1] + (size_t)n * RB + ks * 64u + lane_k * 16u);
                sb[d][q] = (int)S[q >> 1][(size_t)n * GS + ks * 4u + lane_k];
            }
        };
#pragma unroll
        for (unsigned d = 0; d < DEPTH; d++) loadb(d, k0 + d);
        for (unsigned ks = k0; ks < k1; ks += DEPTH) {
#pragma unroll
            for (unsigned d = 0; d < DEPTH; d++) {
                if (ks + d >= k1) break;
                // A: this lane's 32-element group (row lane_n, k-group 4*ks+lane_k), op85's quantizer.
                mpf4_b16 aq = {0u, 0u, 0u, 0u};
                unsigned char sa = PLOW_E8M0_ONE;
                if (xr) {
                    unsigned char tmp[16];
                    mpf4_quant_commit(araw[d], tmp, &sa);
                    __builtin_memcpy(&aq, tmp, 16);
                }
                const moe_sw_i32x8 a = moe_sw_i32x8{(int)aq[0], (int)aq[1], (int)aq[2], (int)aq[3], 0, 0, 0, 0};
#pragma unroll
                for (unsigned q = 0; q < 4; q++) acc[q] = moe_sw_mfma<0>(a, b[d][q], acc[q], (int)sa, sb[d][q]);
                loadb(d, ks + d + DEPTH);
            }
        }
        // Fixed-order cross-wave sum, then GLU + op85's bridge per (row, 32 columns).
#pragma unroll
        for (unsigned q = 0; q < 4; q++)
            *(moe_sw_f32x4*)(red + ((wave * 4u + q) * 64u + lane) * 4u) = acc[q];
        __syncthreads();
        if (wave == 0) {
            moe_sw_f32x4 s[4];
#pragma unroll
            for (unsigned q = 0; q < 4; q++) {
                s[q] = *(const moe_sw_f32x4*)(red + (q * 64u + lane) * 4u);
                for (unsigned w = 1; w < waves; w++) s[q] += *(const moe_sw_f32x4*)(red + ((w * 4u + q) * 64u + lane) * 4u);
            }
#pragma unroll
            for (unsigned h2 = 0; h2 < 2; h2++)
#pragma unroll
                for (unsigned i = 0; i < 4; i++)
                    br[(lane_k * 4u + i) * 32u + h2 * 16u + lane_n] = mpf4_glu(s[h2][i], s[2 + h2][i], act, beta, lbeta);
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
            __builtin_amdgcn_wave_barrier();
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
            if (lane < TM && row_partidx[rowbase + lane] != PLOW_EXPERT_UNUSED) {
                const float* bs = br + lane * 32u;
                float amax = 0.0f;
#pragma unroll
                for (int z = 0; z < 32; z++) amax = fmaxf(amax, fabsf(bs[z]));
                const unsigned char sbv = e8m0_for_amax(amax);
                const float inv = e8m0_inv_f32(sbv);
                mpf4_b16 q;
#pragma unroll
                for (int kk = 0; kk < 4; kk++) {
                    unsigned w = 0u;
#pragma unroll
                    for (int j = 0; j < 8; j++) w |= quant_fp4(bs[kk * 8 + j] * inv) << (j * 4);
                    q[kk] = w;
                }
                mpf4_st16(fu + (size_t)(rowbase + lane) * (I >> 1) + (n0 >> 1), q);
                fu_scale[(size_t)(rowbase + lane) * (I >> 5) + (n0 >> 5)] = sbv;
            }
        }
        __syncthreads();
    }
}
