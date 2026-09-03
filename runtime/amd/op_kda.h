/* op_kda.h — KDA (Kimi Delta Attention), the mixer in 69 of Kimi-K3's 93 layers.
 *
 * Spec: docs/kimi-k3-kda.md. Per head, per token, carrying a state S in [K,V]:
 *
 *     S  <-  (I - beta k k^T) . diag(exp(g)) . S  +  beta k v^T ;      o = S^T q
 *            \___ delta rule ___/  \_ forget gate _/  \_ write _/
 *
 * TWO COMPOSED MEMORY MECHANISMS, and conflating them is the fastest way to a plausible wrong
 * answer:
 *   - the forget gate diag(exp(g)) is UNTARGETED — it decays the whole state, per (head,
 *     key-channel), data-dependent, and bounded to [e^lb, 1) by gate_lower_bound;
 *   - the delta rule (I - beta k k^T) is TARGETED. The kernel L2-normalizes k, so ||k|| = 1 and
 *     this is EXACTLY I minus beta times the orthogonal projector onto k: it erases only the
 *     component of memory stored at key k and leaves everything orthogonal to k untouched.
 *
 * THE STATE IS A DECLARED HBM TENSOR, NOT REGISTERS. [H,D,D] f32 = 6.00 MiB per layer per
 * sequence, CONSTANT in context length (that is the whole architectural argument: 69 KDA layers
 * cost 0.44 GiB at 1M tokens where 24 MLA layers cost 27 GiB). A decode step is a
 * read-modify-write over it, the same kind of object as a KV ring.
 *
 * SLICE MAP FIRST, INNER LOOP SECOND. Every slow kernel in this tree failed the same way:
 * achieved % of ceiling ~= active-workgroup fraction. One workgroup per head is 96/256 = 37.5% at
 * TP1 and 24/256 = 9.4% at TP4 — worse than the MlaMergeFold defect that cost 8.69 ms of a
 * 34.68 ms token. So NOTHING here parallelizes over heads alone.
 *
 * These are the AMD arms. runtime/nvidia/op_mamba.cuh is the precedent to avoid, not to copy: it
 * is monolithic, emitted onto ONE CU, and has no arm in amd/interp.hip at all, so op 90 falls to
 * the silent dispatch default: on gfx950 and computes nothing.
 *
 * SIX ARMS, FOUR ALGORITHMS. Ops 109 and 110 are ops 88 and 102+89 re-sliced into fewer PACKETS,
 * not re-derived: 109 calls 88's own `kda_conv_range` per stream, and 110 is 102's body with one
 * `if constexpr`-shaped branch on where `g` comes from. That is deliberate — a KDA decode layer is
 * launch bound (a packet costs ~12 us in this interpreter, measured; the whole six-op chain's
 * arithmetic is a rounding error against that), so the fusion had to move packets without moving
 * arithmetic. Two bodies computing the same thing is how the transposed-state class of bug gets in.
 *
 * All six ops take (slice, nblk) where a standalone kernel would take (blockIdx.x, gridDim.x) —
 * the interpreter is persistent, grid == CU count, and an op "spread over N workgroups" appears
 * once in the instruction stream and N times in the per-CU streams.
 */
#ifndef PLOW_OP_KDA_H
#define PLOW_OP_KDA_H

#include "amd_common.h"
#include "packed_prefill.h"

#ifndef PLOW_KDA_PF_STATE_RESIDENT
#define PLOW_KDA_PF_STATE_RESIDENT 0
#endif
#ifndef PLOW_KDA_CONV_STEP_DB
#define PLOW_KDA_CONV_STEP_DB 0
#endif

/* Gate activation modes, mirroring [fla]'s `safe_gate` switch (fla/ops/kda/gate.py:118-124). */
enum { PLOW_KDA_GATE_SOFTPLUS = 0, PLOW_KDA_GATE_LOWER_BOUND = 1 };
/* Flag bits for d_kda_state_step. */
enum {
    PLOW_KDA_F_QK_L2NORM = 1u,
    /* THE ROWS ARE INDEPENDENT SEQUENCES, not consecutive tokens of one.
     *
     * A KDA layer's `T` rows mean two different things and the recurrence cannot tell them apart
     * from `T`: on a PREFILL program they are consecutive tokens of ONE sequence and must thread
     * through ONE state; on a BATCHED DECODE program they are B independent sequences, each of
     * which owns its own state. Sharing the state across the second kind runs sequence 1's token
     * into sequence 0's and produces fluent, plausible, WRONG output — no crash, no NaN.
     *
     * So the distinction is carried explicitly. Clear (every program that exists today) makes the
     * per-row state stride 0, the state pointer does not move, and the emitted code is unchanged.
     * See perf-data/archive/k3/k3-batched-decode-design.md §1. */
    PLOW_KDA_F_SEQ_ROWS = 2u,
};

__device__ __forceinline__ uint2 kda_chunk_desc(const uint2* chunks, unsigned chunk, unsigned T) {
    const unsigned row0 = chunk * 64u;
    return chunks ? chunks[chunk] : make_uint2(row0, min(64u, T - row0));
}

__device__ __forceinline__ float kda_sigmoid(float x) { return 1.0f / (1.0f + __expf(-x)); }

/* softplus(x) = log1p(exp(x)), evaluated in the numerically safe branch. [fla] uses
 * `tl.log(1 + tl.exp(x))` guarded to `x` for large x; matching that guard matters because the
 * unbounded gate branch feeds an exp() straight after. */
__device__ __forceinline__ float kda_softplus(float x) {
    return x > 20.0f ? x : __logf(1.0f + __expf(x));
}

/* BT64 chunk-local gate prefix. `chunks[c] = {row0, n_rows}` and n_rows is in [1, 64].
 * One wave owns one (chunk, head, dimension): its lanes are the token axis, so the ordered
 * dependency is a wave shuffle scan rather than a cross-workgroup handoff. The result is stored
 * in log2 units for direct exp2 use by a chunk recurrence. Each chunk deliberately starts from
 * zero; inter-chunk carry is a separate stage. */
__device__ void d_kda_chunk_gate_prefix_bt64(
    float* __restrict__ g_cumsum_log2, float* __restrict__ beta,
    const bf16* __restrict__ g_raw, const bf16* __restrict__ beta_raw,
    const float* __restrict__ A_log, const float* __restrict__ dt_bias,
    const uint2* __restrict__ chunks, unsigned n_chunks, unsigned T, unsigned H, unsigned D,
    unsigned mode, float lb, unsigned slice, unsigned nblk) {
    constexpr float RCP_LN2 = 1.4426950408889634f;
    const unsigned lane = threadIdx.x & (PLOW_WAVE - 1u);
    const unsigned wave = threadIdx.x >> 6;
    const size_t hd_count = (size_t)H * D;
    const size_t n_items = (size_t)n_chunks * hd_count;

    for (size_t item = (size_t)slice * PLOW_WAVES + wave; item < n_items;
         item += (size_t)nblk * PLOW_WAVES) {
        const unsigned chunk = (unsigned)(item / hd_count);
        const unsigned hd = (unsigned)(item - (size_t)chunk * hd_count);
        const unsigned h = hd / D;
        const uint2 desc = kda_chunk_desc(chunks, chunk, T);
        const bool valid = lane < desc.y;
        const size_t row = (size_t)desc.x + lane;

        float gate = 0.0f;
        if (valid) {
            const size_t i = row * hd_count + hd;
            const float a = __expf(A_log[h]);
            const float s = bf2f(g_raw[i]) + dt_bias[hd];
            gate = (mode == PLOW_KDA_GATE_LOWER_BOUND) ? lb * kda_sigmoid(a * s)
                                                       : -a * kda_softplus(s);
        }
        gate *= RCP_LN2;
#pragma unroll
        for (unsigned delta = 1; delta < PLOW_WAVE; delta <<= 1) {
            const float prior = __shfl_up(gate, delta, PLOW_WAVE);
            if (lane >= delta) gate += prior;
        }
        if (valid) {
            g_cumsum_log2[row * hd_count + hd] = gate;
            if (hd % D == 0)
                beta[row * H + h] = kda_sigmoid(bf2f(beta_raw[row * H + h]));
        }
    }
}

/* Dense production preparation: chunk-local gate prefix plus in-place q/k L2 normalization.
 * Null chunk descriptors derive contiguous BT64 chunks and a final tail bounded by T. */
__device__ void d_kda_chunk_prepare_bt64(
    bf16* __restrict__ q, bf16* __restrict__ k, float* __restrict__ g_cumsum_log2,
    float* __restrict__ beta, const bf16* __restrict__ g_raw,
    const bf16* __restrict__ beta_raw, const float* __restrict__ A_log,
    const float* __restrict__ dt_bias, const uint2* __restrict__ chunks, unsigned n_chunks,
    unsigned T, unsigned H, unsigned D, unsigned mode, float lb, unsigned slice, unsigned nblk) {
    d_kda_chunk_gate_prefix_bt64(g_cumsum_log2, beta, g_raw, beta_raw, A_log, dt_bias, chunks,
                                 n_chunks, T, H, D, mode, lb, slice, nblk);
    const unsigned lane = threadIdx.x & 63u, wave = threadIdx.x >> 6;
    const size_t n_rows = (size_t)T * H;
    for (size_t rowh = (size_t)slice * PLOW_WAVES + wave; rowh < n_rows;
         rowh += (size_t)nblk * PLOW_WAVES) {
        const size_t base = rowh * D;
        float qss = 0.0f, kss = 0.0f;
        for (unsigned d = lane; d < D; d += 64u) {
            const float qv = bf2f(q[base + d]), kv = bf2f(k[base + d]);
            qss += qv * qv;
            kss += kv * kv;
        }
        const float qinv = rsqrtf(wave_sum(qss) + 1.0e-6f);
        const float kinv = rsqrtf(wave_sum(kss) + 1.0e-6f);
        for (unsigned d = lane; d < D; d += 64u) {
            q[base + d] = f2bf(bf2f(q[base + d]) * qinv);
            k[base + d] = f2bf(bf2f(k[base + d]) * kinv);
        }
    }
}

/* BC16 diagonal intra-chunk products and solve. One wave owns one
 * (BT64 chunk, 16-token subchunk, head). `g_cumsum_log2` is the chunk-local result above.
 * Products use the native 16x16 bf16 MFMA and accumulate in f32. The strict-lower KK product is
 * multiplied by beta on its row, then solved as (I + L)^-1 in f32. Outputs are [row,H,16], with
 * the local subchunk column in the final dimension. q/k are already normalized and D is a
 * multiple of the MFMA K=32. Inter-subchunk products are deliberately not computed here. */
__device__ void d_kda_chunk_intra_bc16(
    float* __restrict__ Aqk, float* __restrict__ Ainv, const bf16* __restrict__ q,
    const bf16* __restrict__ k, const float* __restrict__ g_cumsum_log2,
    const float* __restrict__ beta, const uint2* __restrict__ chunks, unsigned n_chunks,
    unsigned T, unsigned H, unsigned D, float scale, unsigned slice, unsigned nblk,
    float* __restrict__ lds) {
    const unsigned lane = threadIdx.x & (PLOW_WAVE - 1u);
    const unsigned wave = threadIdx.x >> 6;
    const unsigned token = lane & 15u;
    const unsigned kgroup = lane >> 4;
    const size_t n_items = (size_t)n_chunks * H * 4u;
    float* const mat = lds + (size_t)wave * 16u * 16u;

    for (size_t item = (size_t)slice * PLOW_WAVES + wave; item < n_items;
         item += (size_t)nblk * PLOW_WAVES) {
        const unsigned chunk = (unsigned)(item / ((size_t)H * 4u));
        const unsigned rem = (unsigned)(item - (size_t)chunk * H * 4u);
        const unsigned h = rem >> 2;
        const unsigned sub = rem & 3u;
        const uint2 desc = kda_chunk_desc(chunks, chunk, T);
        const unsigned sub0 = sub * 16u;
        if (sub0 >= desc.y) continue;
        const unsigned valid = min(16u, desc.y - sub0);
        const size_t row0 = (size_t)desc.x + sub0;
        const size_t hd_count = (size_t)H * D;
        const size_t anchor_row = row0 + min(8u, valid - 1u);

        f32x4 qk_acc = (f32x4)(0.0f);
        f32x4 kk_acc = (f32x4)(0.0f);
        for (unsigned d0 = 0; d0 < D; d0 += 32u) {
            bf16x8 qf = (bf16x8)((bf16_t)0);
            bf16x8 kf = (bf16x8)((bf16_t)0);
            bf16x8 kt = (bf16x8)((bf16_t)0);
            const unsigned d = d0 + 8u * kgroup;
            if (token < valid) {
                const size_t row = row0 + token;
#pragma unroll
                for (unsigned j = 0; j < 8; j++) {
                    const size_t i = row * hd_count + (size_t)h * D + d + j;
                    const size_t ia = anchor_row * hd_count + (size_t)h * D + d + j;
                    const float dg = g_cumsum_log2[i] - g_cumsum_log2[ia];
                    qf[j] = __builtin_bit_cast(bf16_t, f2bf(bf2f(q[i]) * exp2f(dg)));
                    kf[j] = __builtin_bit_cast(bf16_t, f2bf(bf2f(k[i]) * exp2f(dg)));
                    kt[j] = __builtin_bit_cast(bf16_t, f2bf(bf2f(k[i]) * exp2f(-dg)));
                }
            }
            qk_acc = plow_mfma_bf16_16x16(qf, kt, qk_acc);
            kk_acc = plow_mfma_bf16_16x16(kf, kt, kk_acc);
        }

#pragma unroll
        for (unsigned e = 0; e < 4; e++) {
            const unsigned m = 4u * kgroup + e;
            const unsigned n = token;
            if (m < valid) {
                const size_t out = ((row0 + m) * H + h) * 16u + n;
                Aqk[out] = n < valid && n <= m ? qk_acc[e] * scale : 0.0f;
            }
            mat[m * 16u + n] = m < valid && n < m ? kk_acc[e] * beta[(row0 + m) * H + h]
                                                   : 0.0f;
        }
        __builtin_amdgcn_wave_barrier();

        for (unsigned i = 0; i < valid; i++) {
            const float row_value = token < valid ? mat[i * 16u + token] : 0.0f;
            float value = token < i ? -row_value : (token == i ? 1.0f : 0.0f);
#pragma unroll
            for (unsigned j = 1; j < 16; j++) {
                const float lij = __shfl(row_value, j, PLOW_WAVE);
                if (j < i && token < j) value -= lij * mat[j * 16u + token];
            }
            __builtin_amdgcn_wave_barrier();
            if (token <= i && lane < 16u) mat[i * 16u + token] = value;
            __builtin_amdgcn_wave_barrier();
        }
        for (unsigned x = lane; x < 16u * 16u; x += PLOW_WAVE) {
            const unsigned m = x >> 4, n = x & 15u;
            if (m < valid)
                Ainv[((row0 + m) * H + h) * 16u + n] = n <= m ? mat[x] : 0.0f;
        }
    }
}

/* Full BT64 intra-chunk matrix. One workgroup owns one (chunk, head); its waves cover the ten
 * lower-triangular BC16 block pairs. All QK/KK products use 16x16 bf16 MFMA with f32
 * accumulation. The complete strict-lower L is then solved in workgroup-local memory, so no
 * inter-workgroup ordering or handoff exists. Outputs are [row,H,64]. Chunk lengths are in
 * [1,64], q/k are already normalized, and D is a multiple of the MFMA K=32. */
__device__ void d_kda_chunk_intra_bt64(
    float* __restrict__ Aqk, float* __restrict__ Ainv, const bf16* __restrict__ q,
    const bf16* __restrict__ k, const float* __restrict__ g_cumsum_log2,
    const float* __restrict__ beta, const uint2* __restrict__ chunks, unsigned n_chunks,
    unsigned T, unsigned H, unsigned D, float scale, unsigned slice, unsigned nblk,
    float* __restrict__ mat) {
    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & (PLOW_WAVE - 1u);
    const unsigned wave = tid >> 6;
    const unsigned token = lane & 15u;
    const unsigned kgroup = lane >> 4;
    const size_t n_items = (size_t)n_chunks * H;

    for (size_t item = slice; item < n_items; item += nblk) {
        const unsigned chunk = (unsigned)(item / H);
        const unsigned h = (unsigned)(item - (size_t)chunk * H);
        const uint2 desc = kda_chunk_desc(chunks, chunk, T);
        const unsigned valid = desc.y;
        const size_t row0 = desc.x;
        const size_t hd_count = (size_t)H * D;

        for (unsigned x = tid; x < 64u * 64u; x += PLOW_THREADS) mat[x] = 0.0f;
        for (unsigned x = tid; x < valid * 64u; x += PLOW_THREADS)
            Aqk[((row0 + x / 64u) * H + h) * 64u + x % 64u] = 0.0f;
        __syncthreads();

        for (unsigned pair = wave; pair < 10u; pair += PLOW_WAVES) {
            const unsigned bm = pair >= 6u ? 3u : (pair >= 3u ? 2u : (pair >= 1u ? 1u : 0u));
            const unsigned bn = pair - bm * (bm + 1u) / 2u;
            const unsigned m0 = bm * 16u, n0 = bn * 16u;
            const unsigned mv = valid > m0 ? min(16u, valid - m0) : 0u;
            const unsigned nv = valid > n0 ? min(16u, valid - n0) : 0u;
            if (mv != 0 && nv != 0) {
                const unsigned anchor_local = bm == bn ? m0 + min(8u, mv - 1u) : m0;
                const size_t anchor_row = row0 + anchor_local;
                f32x4 qk_acc = (f32x4)(0.0f);
                f32x4 kk_acc = (f32x4)(0.0f);
                for (unsigned d0 = 0; d0 < D; d0 += 32u) {
                    bf16x8 qf = (bf16x8)((bf16_t)0);
                    bf16x8 kf = (bf16x8)((bf16_t)0);
                    bf16x8 kt = (bf16x8)((bf16_t)0);
                    const unsigned d = d0 + 8u * kgroup;
                    if (token < mv) {
                        const size_t row = row0 + m0 + token;
#pragma unroll
                        for (unsigned j = 0; j < 8; j++) {
                            const size_t i = row * hd_count + (size_t)h * D + d + j;
                            const size_t ia = anchor_row * hd_count + (size_t)h * D + d + j;
                            const float dg = g_cumsum_log2[i] - g_cumsum_log2[ia];
                            qf[j] = __builtin_bit_cast(bf16_t, f2bf(bf2f(q[i]) * exp2f(dg)));
                            kf[j] = __builtin_bit_cast(bf16_t, f2bf(bf2f(k[i]) * exp2f(dg)));
                        }
                    }
                    if (token < nv) {
                        const size_t row = row0 + n0 + token;
#pragma unroll
                        for (unsigned j = 0; j < 8; j++) {
                            const size_t i = row * hd_count + (size_t)h * D + d + j;
                            const size_t ia = anchor_row * hd_count + (size_t)h * D + d + j;
                            const float dg = g_cumsum_log2[i] - g_cumsum_log2[ia];
                            kt[j] = __builtin_bit_cast(bf16_t, f2bf(bf2f(k[i]) * exp2f(-dg)));
                        }
                    }
                    qk_acc = plow_mfma_bf16_16x16(qf, kt, qk_acc);
                    kk_acc = plow_mfma_bf16_16x16(kf, kt, kk_acc);
                }
#pragma unroll
                for (unsigned e = 0; e < 4; e++) {
                    const unsigned ml = 4u * kgroup + e;
                    const unsigned nl = token;
                    if (ml < mv && nl < nv && (bm != bn || nl <= ml)) {
                        const unsigned m = m0 + ml, n = n0 + nl;
                        Aqk[((row0 + m) * H + h) * 64u + n] = qk_acc[e] * scale;
                        if (n < m)
                            mat[m * 64u + n] =
                                kk_acc[e] * beta[(row0 + m) * H + h];
                    }
                }
            }
        }
        __syncthreads();

        if (wave == 0) {
            for (unsigned i = 0; i < valid; i++) {
                const float row_value = mat[i * 64u + lane];
                float value = lane < i ? -row_value : (lane == i ? 1.0f : 0.0f);
#pragma unroll 1
                for (unsigned j = 1; j < 64u; j++) {
                    const float lij = __shfl(row_value, j, PLOW_WAVE);
                    if (j < i && lane < j) value -= lij * mat[j * 64u + lane];
                }
                __builtin_amdgcn_wave_barrier();
                if (lane <= i) mat[i * 64u + lane] = value;
                __builtin_amdgcn_wave_barrier();
            }
        }
        __syncthreads();
        for (unsigned x = tid; x < valid * 64u; x += PLOW_THREADS) {
            const unsigned m = x / 64u, n = x % 64u;
            Ainv[((row0 + m) * H + h) * 64u + n] = n <= m ? mat[m * 64u + n] : 0.0f;
        }
        __syncthreads();
    }
}

/* Transform a full BT64 inverse into the write/value factors used by the chunk carry:
 *   W = Ainv @ (beta * K * exp2(g)), U = Ainv @ (beta * V).
 * One wave owns one (chunk, head, 16 token rows, 16 output channels) tile. */
__device__ void d_kda_chunk_wu_bt64(
    bf16* __restrict__ W, bf16* __restrict__ U, bf16* __restrict__ q,
    const float* __restrict__ Ainv, const bf16* __restrict__ k, const bf16* __restrict__ v,
    const float* __restrict__ g_cumsum_log2, const float* __restrict__ beta,
    const uint2* __restrict__ chunks, unsigned n_chunks, unsigned T, unsigned H, unsigned D,
    unsigned V, float scale, bool q_precompute, unsigned slice, unsigned nblk) {
    const unsigned lane = threadIdx.x & 63u, wave = threadIdx.x >> 6;
    const unsigned col = lane & 15u, kgroup = lane >> 4;
    const unsigned dtiles = (D + 15u) / 16u, vtiles = (V + 15u) / 16u;
    const unsigned otiles = dtiles + vtiles;
    if (q_precompute) {
        const size_t q_items = (size_t)T * H * D;
        for (size_t qi = (size_t)slice * PLOW_THREADS + threadIdx.x; qi < q_items;
             qi += (size_t)nblk * PLOW_THREADS) {
            const bf16 qs = f2bf(bf2f(q[qi]) * scale);
            q[qi] = f2bf(bf2f(qs) * exp2f(g_cumsum_log2[qi]));
        }
    }
    const size_t n_items = (size_t)n_chunks * H * 4u * otiles;
    for (size_t item = (size_t)slice * PLOW_WAVES + wave; item < n_items;
         item += (size_t)nblk * PLOW_WAVES) {
        size_t z = item;
        const unsigned ot = z % otiles; z /= otiles;
        const unsigned mb = z & 3u; z >>= 2;
        const unsigned h = z % H, chunk = z / H;
        const uint2 desc = kda_chunk_desc(chunks, chunk, T);
        const unsigned m0 = mb * 16u;
        const unsigned mv = desc.y > m0 ? min(16u, desc.y - m0) : 0u;
        if (mv == 0) continue;
        const bool make_w = ot < dtiles;
        const unsigned out0 = (make_w ? ot : ot - dtiles) * 16u;
        const unsigned width = make_w ? D : V;
        const size_t row0 = desc.x;
        f32x4 acc = (f32x4)(0.0f);
        for (unsigned s0 = 0; s0 < 64u; s0 += 32u) {
            bf16x8 af = (bf16x8)((bf16_t)0), bf = (bf16x8)((bf16_t)0);
            const unsigned s = s0 + 8u * kgroup;
#pragma unroll
            for (unsigned j = 0; j < 8; j++) {
                const unsigned sl = s + j;
                if (col < mv && sl < desc.y) {
                    const size_t ai = ((row0 + m0 + col) * H + h) * 64u + sl;
                    af[j] = __builtin_bit_cast(bf16_t, f2bf(Ainv[ai]));
                }
                if (out0 + col < width && sl < desc.y) {
                    const size_t row = row0 + sl;
                    const float b = beta[row * H + h];
                    float x;
                    if (make_w) {
                        const size_t i = (row * H + h) * D + out0 + col;
                        x = bf2f(k[i]) * b * exp2f(g_cumsum_log2[i]);
                    } else {
                        const size_t i = (row * H + h) * V + out0 + col;
                        x = bf2f(v[i]) * b;
                    }
                    bf[j] = __builtin_bit_cast(bf16_t, f2bf(x));
                }
            }
            acc = plow_mfma_bf16_16x16(af, bf, acc);
        }
#pragma unroll
        for (unsigned e = 0; e < 4; e++) {
            const unsigned ml = 4u * kgroup + e;
            if (ml < mv && out0 + col < width) {
                const size_t row = row0 + m0 + ml;
                const size_t oi = make_w ? (row * H + h) * D + out0 + col
                                         : (row * H + h) * V + out0 + col;
                (make_w ? W : U)[oi] = f2bf(acc[e]);
            }
        }
    }
}

/* Ordered multi-chunk carry for one sequence (chunk lengths [1,64], D <= 128 and divisible by
 * 32). One workgroup owns a (head, V16) state tile and
 * walks `chunks` in order; distinct workgroups own disjoint V rows, so no cross-workgroup handoff
 * is required. The decomposition matches FLA: V' = U-WH, O = Q exp(g) H + Aqk V', then
 * H = H exp(g_last) + V'^T (K exp(g_last-g)). State is V-first f32. */
__device__ void d_kda_chunk_carry_bt64(
    bf16* __restrict__ out, float* __restrict__ state, const bf16* __restrict__ q,
    const bf16* __restrict__ k, const bf16* __restrict__ W, const bf16* __restrict__ U,
    const float* __restrict__ Aqk, const float* __restrict__ g_cumsum_log2,
    const uint2* __restrict__ chunks, unsigned n_chunks, unsigned T, unsigned H, unsigned D,
    unsigned V, float scale, unsigned slice, unsigned nblk, float* __restrict__ st,
    bf16* __restrict__ vsm, float* __restrict__ osm, bool q_precomputed = false) {
    const unsigned tid = threadIdx.x, lane = tid & 63u, wave = tid >> 6;
    const unsigned token = lane & 15u, kgroup = lane >> 4;
    const unsigned vtiles = (V + 15u) / 16u;
    const size_t n_items = (size_t)H * vtiles;
    for (size_t item = slice; item < n_items; item += nblk) {
        const unsigned h = (unsigned)(item / vtiles), vt = (unsigned)(item % vtiles);
        const unsigned v0 = vt * 16u, vv = min(16u, V - v0);
        for (unsigned x = tid; x < vv * D; x += PLOW_THREADS)
            st[x] = state[((size_t)h * V + v0 + x / D) * D + x % D];
        __syncthreads();
        for (unsigned c = 0; c < n_chunks; c++) {
            const uint2 desc = kda_chunk_desc(chunks, c, T);
            const size_t row0 = desc.x;
            for (unsigned mb = wave; mb < 4u; mb += PLOW_WAVES) {
                const unsigned m0 = mb * 16u;
                const unsigned mv = desc.y > m0 ? min(16u, desc.y - m0) : 0u;
                if (mv == 0) continue;
                f32x4 pred = (f32x4)(0.0f), from_state = (f32x4)(0.0f);
                for (unsigned d0 = 0; d0 < D; d0 += 32u) {
                    bf16x8 wf = (bf16x8)((bf16_t)0), qf = (bf16x8)((bf16_t)0);
                    bf16x8 sf = (bf16x8)((bf16_t)0);
                    const unsigned d = d0 + 8u * kgroup;
                    if (token < mv) {
                        const size_t row = row0 + m0 + token;
#pragma unroll
                        for (unsigned j = 0; j < 8; j++) {
                            const size_t i = (row * H + h) * D + d + j;
                            wf[j] = __builtin_bit_cast(bf16_t, W[i]);
                            if (q_precomputed) {
                                qf[j] = __builtin_bit_cast(bf16_t, q[i]);
                            } else {
                                const float qs = bf2f(f2bf(bf2f(q[i]) * scale));
                                qf[j] = __builtin_bit_cast(
                                    bf16_t, f2bf(qs * exp2f(g_cumsum_log2[i])));
                            }
                        }
                    }
                    if (token < vv) {
#pragma unroll
                        for (unsigned j = 0; j < 8; j++)
                            sf[j] = __builtin_bit_cast(bf16_t, f2bf(st[token * D + d + j]));
                    }
                    pred = plow_mfma_bf16_16x16(wf, sf, pred);
                    from_state = plow_mfma_bf16_16x16(qf, sf, from_state);
                }
#pragma unroll
                for (unsigned e = 0; e < 4; e++) {
                    const unsigned ml = 4u * kgroup + e;
                    if (ml < mv && token < vv) {
                        const size_t row = row0 + m0 + ml;
                        vsm[(m0 + ml) * 16u + token] =
                            f2bf(bf2f(U[(row * H + h) * V + v0 + token]) - pred[e]);
                        osm[(m0 + ml) * 16u + token] = from_state[e];
                    }
                }
            }
            __syncthreads();
            for (unsigned mb = wave; mb < 4u; mb += PLOW_WAVES) {
                const unsigned m0 = mb * 16u;
                const unsigned mv = desc.y > m0 ? min(16u, desc.y - m0) : 0u;
                if (mv == 0) continue;
                f32x4 local = (f32x4)(0.0f);
                for (unsigned s0 = 0; s0 < 64u; s0 += 32u) {
                    bf16x8 af = (bf16x8)((bf16_t)0), vf = (bf16x8)((bf16_t)0);
                    const unsigned s = s0 + 8u * kgroup;
#pragma unroll
                    for (unsigned j = 0; j < 8; j++) {
                        if (token < mv && s + j < desc.y) {
                            const size_t ai = ((row0 + m0 + token) * H + h) * 64u + s + j;
                            af[j] = __builtin_bit_cast(bf16_t, f2bf(Aqk[ai]));
                        }
                        if (token < vv && s + j < desc.y)
                            vf[j] = __builtin_bit_cast(bf16_t, vsm[(s + j) * 16u + token]);
                    }
                    local = plow_mfma_bf16_16x16(af, vf, local);
                }
#pragma unroll
                for (unsigned e = 0; e < 4; e++) {
                    const unsigned ml = 4u * kgroup + e;
                    if (ml < mv && token < vv) {
                        const size_t row = row0 + m0 + ml;
                        out[(row * H + h) * V + v0 + token] =
                            f2bf(osm[(m0 + ml) * 16u + token] + local[e]);
                    }
                }
            }
            __syncthreads();

            const size_t last = row0 + desc.y - 1u;
            for (unsigned d0 = wave * 16u; d0 < D; d0 += PLOW_WAVES * 16u) {
                f32x4 upd = (f32x4)(0.0f);
                for (unsigned s0 = 0; s0 < 64u; s0 += 32u) {
                    bf16x8 vf = (bf16x8)((bf16_t)0), kf = (bf16x8)((bf16_t)0);
                    bf16x8 kr = (bf16x8)((bf16_t)0);
                    const unsigned s = s0 + 8u * kgroup;
#pragma unroll
                    for (unsigned j = 0; j < 8; j++) {
                        if (token < vv && s + j < desc.y)
                            vf[j] = __builtin_bit_cast(bf16_t, vsm[(s + j) * 16u + token]);
                        if (d0 + token < D && s + j < desc.y) {
                            const size_t row = row0 + s + j;
                            const size_t i = (row * H + h) * D + d0 + token;
                            const size_t il = (last * H + h) * D + d0 + token;
                            const float scaled =
                                bf2f(k[i]) * exp2f(g_cumsum_log2[il] - g_cumsum_log2[i]);
                            const bf16 high = f2bf(scaled);
                            kf[j] = __builtin_bit_cast(bf16_t, high);
                            kr[j] = __builtin_bit_cast(bf16_t, f2bf(scaled - bf2f(high)));
                        }
                    }
                    upd = plow_mfma_bf16_16x16(vf, kf, upd);
                    upd = plow_mfma_bf16_16x16(vf, kr, upd);
                }
#pragma unroll
                for (unsigned e = 0; e < 4; e++) {
                    const unsigned vl = 4u * kgroup + e;
                    if (vl < vv && d0 + token < D) {
                        const size_t il = (last * H + h) * D + d0 + token;
                        st[vl * D + d0 + token] =
                            st[vl * D + d0 + token] * exp2f(g_cumsum_log2[il]) + upd[e];
                    }
                }
            }
            __syncthreads();
        }
        for (unsigned x = tid; x < vv * D; x += PLOW_THREADS)
            state[((size_t)h * V + v0 + x / D) * D + x % D] = st[x];
        __syncthreads();
    }
}

#if PLOW_PACKED_PREFILL_KDA_CONSUMERS
__device__ void d_kda_chunk_prepare_packed_bt64(
    bf16* q, bf16* k, float* g_prefix, float* beta, const bf16* g_raw,
    const bf16* beta_raw, const float* A_log, const float* dt_bias, unsigned H, unsigned D,
    unsigned mode, float lb, unsigned slice, unsigned nblk, const PlowProgram* prog) {
    for (unsigned si = 0; si < prog->n_prefill_spans; ++si) {
        const PlowPrefillSpan* span = plow_packed_prefill_span(prog, si);
        const size_t rd = (size_t)span->row0 * H * D;
        const size_t rh = (size_t)span->row0 * H;
        d_kda_chunk_prepare_bt64(q + rd, k + rd, g_prefix + rd, beta + rh, g_raw + rd,
                                 beta_raw + rh, A_log, dt_bias, nullptr,
                                 (span->n_rows + 63u) / 64u, span->n_rows, H, D, mode, lb,
                                 slice, nblk);
    }
}

__device__ void d_kda_chunk_intra_packed_bt64(
    float* Aqk, float* Ainv, const bf16* q, const bf16* k, const float* g_prefix,
    const float* beta, unsigned H, unsigned D, float scale, unsigned slice, unsigned nblk,
    float* mat, const PlowProgram* prog) {
    for (unsigned si = 0; si < prog->n_prefill_spans; ++si) {
        const PlowPrefillSpan* span = plow_packed_prefill_span(prog, si);
        const size_t rd = (size_t)span->row0 * H * D;
        const size_t rh = (size_t)span->row0 * H;
        const size_t ra = (size_t)span->row0 * H * 64u;
        d_kda_chunk_intra_bt64(Aqk + ra, Ainv + ra, q + rd, k + rd, g_prefix + rd,
                               beta + rh, nullptr, (span->n_rows + 63u) / 64u,
                               span->n_rows, H, D, scale, slice, nblk, mat);
    }
}

__device__ void d_kda_chunk_wu_packed_bt64(
    bf16* W, bf16* U, bf16* q, const float* Ainv, const bf16* k, const bf16* v,
    const float* g_prefix, const float* beta, unsigned H, unsigned D, unsigned V,
    float scale, bool q_precompute, unsigned slice, unsigned nblk, const PlowProgram* prog) {
    for (unsigned si = 0; si < prog->n_prefill_spans; ++si) {
        const PlowPrefillSpan* span = plow_packed_prefill_span(prog, si);
        const size_t rd = (size_t)span->row0 * H * D;
        const size_t rv = (size_t)span->row0 * H * V;
        const size_t rh = (size_t)span->row0 * H;
        const size_t ra = (size_t)span->row0 * H * 64u;
        d_kda_chunk_wu_bt64(W + rd, U + rv, q ? q + rd : nullptr, Ainv + ra, k + rd,
                            v + rv, g_prefix + rd, beta + rh, nullptr,
                            (span->n_rows + 63u) / 64u, span->n_rows, H, D, V, scale,
                            q_precompute, slice, nblk);
    }
}

__device__ void d_kda_chunk_carry_packed_bt64(
    bf16* out, float* state, const bf16* q, const bf16* k, const bf16* W,
    const bf16* U, const float* Aqk, const float* g_prefix, unsigned H, unsigned D,
    unsigned V, float scale, unsigned slice, unsigned nblk, float* st, bf16* vsm,
    float* osm, const PlowProgram* prog, bool q_precomputed = false) {
    const unsigned vtiles = (V + 15u) / 16u;
    for (unsigned si = 0; si < prog->n_prefill_spans; ++si) {
        const PlowPrefillSpan* span = plow_packed_prefill_span(prog, si);
        float* ss = state + (size_t)span->state_slot * H * V * D;
        if (span->flags & PLOW_PREFILL_SPAN_RESET_STATE) {
            for (size_t item = slice; item < (size_t)H * vtiles; item += nblk) {
                const unsigned vt = (unsigned)(item % vtiles);
                const unsigned vv = min(16u, V - vt * 16u);
                const size_t base = ((item / vtiles) * V + vt * 16u) * D;
                for (unsigned x = threadIdx.x; x < vv * D; x += PLOW_THREADS) ss[base + x] = 0.0f;
                __syncthreads();
            }
        }
        const size_t rd = (size_t)span->row0 * H * D;
        const size_t rv = (size_t)span->row0 * H * V;
        const size_t ra = (size_t)span->row0 * H * 64u;
        d_kda_chunk_carry_bt64(out + rv, ss, q + rd, k + rd, W + rd, U + rv, Aqk + ra,
                               g_prefix + rd, nullptr, (span->n_rows + 63u) / 64u,
                               span->n_rows, H, D, V, scale, slice, nblk, st, vsm, osm,
                               q_precomputed);
    }
}
#endif

/* -------------------------------------------------------------------------------------------
 * op 88 — KDA short conv.
 *
 * ONE stream's causal depthwise convolution of width W over `conv_dim` channels, then an
 * activation. KDA has three such streams (q, k, v); this arm takes one, and op 109 takes all three
 * in one packet. `groups = hidden_size` makes it depthwise, `padding = W-1` makes it causal, and
 * there is no bias (the checkpoint ships no *_conv1d.bias). This is what gives KDA local W-token
 * mixing that a pure linear-attention recurrence cannot express — it is 0.03% of the layer's MACs
 * and it is not optional.
 *
 * `state` is the rolling input window, [conv_dim, W] f32, holding the last W inputs per channel
 * with the CURRENT token at slot W-1:
 *      state.roll(-1); state[:, W-1] = x_t; y = sum_j state[:, j] * w[:, j]
 * ([fla] short_conv.py:232-235). [vllm] instead keeps W-1 slots and prepends the current token;
 * both are correct and they differ by 36864 elements per layer. This is the [fla] convention
 * because the reference the numeric gate runs against is [fla].
 *
 * SLICE MAP: the conv is a W-tap STENCIL, not a scan — y_t depends on x_{t-W+1..t}, never on
 * y_{t-1} — so prefill is parallel over (t, channel). Only its first W-1 rows read the incoming
 * window; one designated worker per channel computes those rows and publishes the final window,
 * while the remaining rows are spread across every worker in the channel's packet slice. Decode
 * and independent-sequence rows retain the serial per-channel path below.
 *
 * Each block takes a CONTIGUOUS chunk of channels rather than a strided one: a channel's window is
 * 4 contiguous f32 = one global_load_dwordx4, and contiguous chunks keep the 512-byte-per-16-lane
 * coalescing while spreading the work over all 256 CUs. At conv_dim = 36864 that is 144 channels
 * per block — 144 of 512 lanes busy, which is a wave-level idle, not a CU-level one, and CU spread
 * is what the bandwidth wants.
 */
/* One stream's channels [c0, c1) of the conv, for THIS workgroup. Factored out so op 88 and op 109
 * share ONE body: the fused arm calls this on a per-stream sub-range, so "the fused conv equals
 * three separate convs" is true by construction and not by tolerance. `conv_dim` is the stream's
 * own channel stride, which is what makes the split legal — nothing in the loop couples channels. */
__device__ __forceinline__ void kda_conv_range(bf16* __restrict__ out, const bf16* __restrict__ x,
                                               const float* __restrict__ w,
                                               float* __restrict__ state, unsigned T,
                                               unsigned conv_dim, unsigned W, unsigned act,
                                               unsigned c0, unsigned c1, size_t bstride,
                                               const unsigned* __restrict__ parked) {
    /* INDEPENDENT-SEQUENCE PATH. `bstride != 0` means the T rows are B separate sequences
     * (batched decode), so each row owns its own sliding window: load it, roll ONE token
     * through, store it back. The shared path below is the opposite and is the one every
     * program uses today — it loads the window once, rolls all T consecutive tokens of ONE
     * sequence through it, and stores once, which is the whole point of a conv state.
     *
     * Kept as a separate loop rather than a stride inside the shared one because the LOAD and
     * STORE move, not just the address: hoisting them out of the token loop is exactly what
     * makes the shared path correct, and exactly what makes it wrong for independent rows. */
    if (bstride) {
        for (unsigned c = c0 + threadIdx.x; c < c1; c += PLOW_THREADS) {
            enum { PLOW_KDA_WMAX_B = 8 };
            const unsigned Wc = W < PLOW_KDA_WMAX_B ? W : PLOW_KDA_WMAX_B;
            for (unsigned t = 0; t < T; t++) {
                /* Same contract as the recurrence's mask: a parked row must not have its
                 * convolution window shifted, or the sequence it belongs to resumes against a
                 * window holding tokens from nobody. */
                if (parked && parked[t]) continue;
                float* st = state + (size_t)t * bstride + (size_t)c * W;
                float win[PLOW_KDA_WMAX_B], tap[PLOW_KDA_WMAX_B];
#pragma unroll
                for (unsigned j = 0; j < PLOW_KDA_WMAX_B; j++) {
                    win[j] = j < Wc ? st[j] : 0.0f;
                    tap[j] = j < Wc ? w[(size_t)c * W + j] : 0.0f;
                }
#pragma unroll
                for (unsigned j = 0; j + 1 < PLOW_KDA_WMAX_B; j++) win[j] = win[j + 1];
                win[Wc - 1] = bf2f(x[(size_t)t * conv_dim + c]);
                float y = 0.0f;
#pragma unroll
                for (unsigned j = 0; j < PLOW_KDA_WMAX_B; j++) y += win[j] * tap[j];
                st_act1(&out[(size_t)t * conv_dim + c], f2bf(act == 1u ? act_silu(y) : y));
#pragma unroll
                for (unsigned j = 0; j < PLOW_KDA_WMAX_B; j++)
                    if (j < Wc) st[j] = win[j];
            }
        }
        return;
    }
    for (unsigned c = c0 + threadIdx.x; c < c1; c += PLOW_THREADS) {
        /* The window and the taps. W is 4 for K3 and the loop bound is a runtime value, so this is
         * written as a small fixed array; PLOW_KDA_WMAX bounds the register cost. */
        enum { PLOW_KDA_WMAX = 8 };
        float win[PLOW_KDA_WMAX], tap[PLOW_KDA_WMAX];
        const unsigned Wc = W < PLOW_KDA_WMAX ? W : PLOW_KDA_WMAX;
#pragma unroll
        for (unsigned j = 0; j < PLOW_KDA_WMAX; j++) {
            win[j] = j < Wc ? state[(size_t)c * W + j] : 0.0f;
            tap[j] = j < Wc ? w[(size_t)c * W + j] : 0.0f;
        }
        for (unsigned t = 0; t < T; t++) {
            /* roll left, insert x_t at the newest slot */
#pragma unroll
            for (unsigned j = 0; j + 1 < PLOW_KDA_WMAX; j++) win[j] = win[j + 1];
            win[Wc - 1] = bf2f(x[(size_t)t * conv_dim + c]);
            float y = 0.0f;
#pragma unroll
            for (unsigned j = 0; j < PLOW_KDA_WMAX; j++) y += win[j] * tap[j];
            /* activation AFTER the convolution (short_conv.py:55-72) */
            st_act1(&out[(size_t)t * conv_dim + c], f2bf(act == 1u ? act_silu(y) : y));
        }
#pragma unroll
        for (unsigned j = 0; j < PLOW_KDA_WMAX; j++)
            if (j < Wc) state[(size_t)c * W + j] = win[j];
    }
}

__device__ void d_kda_conv(bf16* __restrict__ out, const bf16* __restrict__ x,
                           const float* __restrict__ w, float* __restrict__ state, unsigned T,
                           unsigned conv_dim, unsigned W, unsigned act, unsigned slice,
                           unsigned nblk, size_t bstride) {
    const unsigned chunk = (conv_dim + nblk - 1) / nblk;
    const unsigned c0 = slice * chunk;
    unsigned c1 = c0 + chunk;
    if (c1 > conv_dim) c1 = conv_dim;
    if (c0 < c1) kda_conv_range(out, x, w, state, T, conv_dim, W, act, c0, c1, bstride, nullptr);
}

/* -------------------------------------------------------------------------------------------
 * op 109 — the same conv over all THREE streams in one packet.
 *
 * WHY, given that op 88's own note argues three packets is three times the concurrency: because
 * at batch 1 that is not what three packets buys. `runtime/tests/kda_fuse_bench_gfx950.c` measures
 * a packet in this interpreter at ~12 us against a KDA chain whose entire arithmetic is a rounding
 * error — 414 packets of the six-op chain cost 5.03 ms over 69 layers at TP8, and the cost is
 * LINEAR in the packet count with a slope of 12.08 us and an intercept of 0.02 ms. Three
 * independent packets therefore cost three times one packet holding the same work. The
 * concurrency op 88 was protecting is real and is preserved here; what is deleted is two counter
 * gates per layer, 138 per token.
 *
 * THE MERGE IS ALONG THE OUTPUT AXIS. The block still takes a CONTIGUOUS chunk, now of the 3*C
 * concatenated channel axis, so per CU the channel count RISES (48 -> 144 at TP1, 6 -> 18 at TP8)
 * on the same 256 CUs. Nothing that ran in parallel starts running in sequence. That is the
 * `GemvQkvg` direction; `GLM_GROUP=1`, which collapsed disjoint CU slices into a loop for +2.88 ms,
 * is the other one.
 *
 * A chunk of the 3*C axis crosses at most two stream boundaries, so this is a 3-iteration loop
 * over intersections, each delegating to op 88's own body. The streams keep
 * SEPARATE buffers — nothing here assumes q|k|v are contiguous in memory, which they are not:
 * `GemvQkvg` writes three distinct handles.
 */
__device__ void d_kda_conv3(bf16* __restrict__ oq, bf16* __restrict__ ok, bf16* __restrict__ ov,
                            const bf16* __restrict__ xq, const bf16* __restrict__ xk,
                            const bf16* __restrict__ xv, const float* __restrict__ wq,
                            const float* __restrict__ wk, const float* __restrict__ wv,
                            float* __restrict__ sq, float* __restrict__ sk,
                            float* __restrict__ sv, unsigned T, unsigned C, unsigned W,
                            unsigned act, unsigned slice, unsigned nblk, size_t bstride,
                            const unsigned* __restrict__ parked) {
    const unsigned total = 3u * C;
    const unsigned chunk = (total + nblk - 1) / nblk;
    const unsigned g0 = slice * chunk;
    unsigned g1 = g0 + chunk;
    if (g1 > total) g1 = total;
    if (g0 >= g1) return;
    if (T > 1u && bstride == 0) {
        enum { PLOW_KDA_WMAX = 8 };
        const unsigned Wc = W < PLOW_KDA_WMAX ? W : PLOW_KDA_WMAX;
        if (Wc == 0) __builtin_trap();

        /* Rows [0,W-1) still read the incoming convolution window. Give all of them, and the
         * final-window store, to one worker per (stream,channel): no other worker reads `state`, so
         * publishing the new window needs no grid-wide barrier. */
        const unsigned prefix = T < Wc - 1u ? T : Wc - 1u;
        for (unsigned g = g0 + threadIdx.x; g < g1; g += PLOW_THREADS) {
            const unsigned s = g / C;
            const unsigned c = g - s * C;
            bf16* out = s == 0 ? oq : (s == 1 ? ok : ov);
            const bf16* x = s == 0 ? xq : (s == 1 ? xk : xv);
            const float* w = s == 0 ? wq : (s == 1 ? wk : wv);
            float* state = s == 0 ? sq : (s == 1 ? sk : sv);
            float win[PLOW_KDA_WMAX], tap[PLOW_KDA_WMAX];
#pragma unroll
            for (unsigned j = 0; j < PLOW_KDA_WMAX; ++j) {
                win[j] = j < Wc ? state[(size_t)c * W + j] : 0.0f;
                tap[j] = j < Wc ? w[(size_t)c * W + j] : 0.0f;
            }
            for (unsigned t = 0; t < prefix; ++t) {
#pragma unroll
                for (unsigned j = 0; j + 1 < PLOW_KDA_WMAX; ++j) win[j] = win[j + 1];
                win[Wc - 1u] = bf2f(x[(size_t)t * C + c]);
                float y = 0.0f;
#pragma unroll
                for (unsigned j = 0; j < PLOW_KDA_WMAX; ++j) y += win[j] * tap[j];
                st_act1(&out[(size_t)t * C + c], f2bf(act == 1u ? act_silu(y) : y));
            }
#pragma unroll
            for (unsigned j = 0; j < PLOW_KDA_WMAX; ++j) {
                if (j >= Wc) continue;
                const float v = j + T < Wc
                                    ? win[j]
                                    : bf2f(x[((size_t)T - Wc + j) * C + c]);
                state[(size_t)c * W + j] = v;
            }
        }

        /* From row W-1 onward every tap comes from this chunk's input. Split that row range into
         * contiguous spans per channel, so all workers participate while each reuses one tap vector
         * and rolls one local input window. */
        const unsigned row0 = Wc - 1u;
        if (row0 >= T) return;
        const unsigned rows = T - row0;
        const unsigned nch = g1 - g0;
        const unsigned ncpar = nch < PLOW_THREADS ? nch : PLOW_THREADS;
        const unsigned cworker = threadIdx.x % ncpar;
        const unsigned lane = threadIdx.x / ncpar;
        const unsigned lanes = (PLOW_THREADS + ncpar - 1u - cworker) / ncpar;
        const unsigned rchunk = (rows + lanes - 1u) / lanes;
        const unsigned t0 = row0 + lane * rchunk;
        unsigned t1 = t0 + rchunk;
        if (t1 > T) t1 = T;
        for (unsigned g = g0 + cworker; g < g1; g += ncpar) {
            if (t0 >= t1) continue;
            const unsigned s = g / C;
            const unsigned c = g - s * C;
            bf16* out = s == 0 ? oq : (s == 1 ? ok : ov);
            const bf16* x = s == 0 ? xq : (s == 1 ? xk : xv);
            const float* w = s == 0 ? wq : (s == 1 ? wk : wv);
            float win[PLOW_KDA_WMAX], tap[PLOW_KDA_WMAX];
#pragma unroll
            for (unsigned j = 0; j < PLOW_KDA_WMAX; ++j) {
                win[j] = j < Wc
                             ? bf2f(x[((size_t)t0 - (Wc - 1u - j)) * C + c])
                             : 0.0f;
                tap[j] = j < Wc ? w[(size_t)c * W + j] : 0.0f;
            }
            for (unsigned t = t0; t < t1; ++t) {
                float y = 0.0f;
#pragma unroll
                for (unsigned j = 0; j < PLOW_KDA_WMAX; ++j) y += win[j] * tap[j];
                st_act1(&out[(size_t)t * C + c], f2bf(act == 1u ? act_silu(y) : y));
                if (t + 1u < t1) {
#pragma unroll
                    for (unsigned j = 0; j + 1u < PLOW_KDA_WMAX; ++j) {
                        win[j] = win[j + 1u];
                    }
                    win[Wc - 1u] = bf2f(x[((size_t)t + 1u) * C + c]);
                }
            }
        }
        return;
    }
    /* ROLLED, not unrolled. `kda_conv_range` is force-inlined and holds 2*PLOW_KDA_WMAX floats of
     * window and taps, so unrolling puts three copies of that in the function; rolled there is ONE
     * inline site and the pointer triples become selects. Measured on the K3 decode object it
     * changes NOTHING — 254 VGPR / occ 2 / 4 spill either way — so this is a code-size choice, not
     * a register one, and it is written down that way rather than claimed as a win. (The 4 spilled
     * VGPRs are not this op's: adding EITHER new arm alone produces them, and `noinline` on both
     * does not remove them. The K3=0 object is unchanged at 0.) */
#pragma unroll 1
    for (unsigned s = 0; s < 3u; s++) {
        const unsigned lo = s * C;
        const unsigned a = g0 > lo ? g0 - lo : 0u;         /* stream-local start */
        const unsigned bb = g1 > lo ? g1 - lo : 0u;        /* stream-local end   */
        const unsigned b = bb > C ? C : bb;
        if (a >= b) continue;
        bf16* o = s == 0 ? oq : (s == 1 ? ok : ov);
        const bf16* x = s == 0 ? xq : (s == 1 ? xk : xv);
        const float* w = s == 0 ? wq : (s == 1 ? wk : wv);
        float* st = s == 0 ? sq : (s == 1 ? sk : sv);
        kda_conv_range(o, x, w, st, T, C, W, act, a, b, bstride, parked);
    }
}

__device__ void d_kda_conv3_packed(
    bf16* oq, bf16* ok, bf16* ov, const bf16* xq, const bf16* xk, const bf16* xv,
    const float* wq, const float* wk, const float* wv, float* sq, float* sk, float* sv,
    unsigned C, unsigned W, unsigned act, unsigned slice, unsigned nblk,
    const PlowProgram* prog) {
    const unsigned total = 3u * C;
    const unsigned chunk = (total + nblk - 1u) / nblk;
    const unsigned g0 = slice * chunk;
    const unsigned g1 = g0 + chunk < total ? g0 + chunk : total;
    for (unsigned si = 0; si < prog->n_prefill_spans; ++si) {
        const PlowPrefillSpan* span = plow_packed_prefill_span(prog, si);
        float* states[3] = {sq + (size_t)span->state_slot * C * W,
                            sk + (size_t)span->state_slot * C * W,
                            sv + (size_t)span->state_slot * C * W};
        if (span->flags & PLOW_PREFILL_SPAN_RESET_STATE) {
            for (unsigned g = g0 + threadIdx.x; g < g1; g += PLOW_THREADS) {
                const unsigned stream = g / C, channel = g % C;
                for (unsigned j = 0; j < W; ++j)
                    states[stream][(size_t)channel * W + j] = 0.0f;
            }
            __syncthreads();
        }
        d_kda_conv3(oq + (size_t)span->row0 * C, ok + (size_t)span->row0 * C,
                    ov + (size_t)span->row0 * C, xq + (size_t)span->row0 * C,
                    xk + (size_t)span->row0 * C, xv + (size_t)span->row0 * C,
                    wq, wk, wv, states[0], states[1], states[2], span->n_rows, C, W, act,
                    slice, nblk, 0u, nullptr);
    }
}

/* -------------------------------------------------------------------------------------------
 * op 89 — KDA gate pre-pass. Pure elementwise over [T,H,D] plus [T,H].
 *
 *   mode 1 (K3, gate_lower_bound = -5.0):
 *       g[t,h,d] = lb * sigmoid( exp(A_log[h]) * (g_raw[t,h,d] + dt_bias[h,d]) )
 *   mode 0 (Kimi-Linear and every released vLLM):
 *       g[t,h,d] = -exp(A_log[h]) * softplus( g_raw[t,h,d] + dt_bias[h,d] )
 *   both:   beta[t,h] = sigmoid( beta_raw[t,h] )
 *
 * The bounded branch clamps g to [lb, 0), so the per-step decay exp(g) lies in (e^lb, 1): the
 * state can never be zeroed by the gate in one step and can never grow. K3 is the FIRST
 * checkpoint to ship it — do not expect the Kimi Linear paper, or any released vLLM, to describe
 * it, and do not use them as an oracle for it.
 *
 * A_log is indexed PER HEAD; dt_bias is [H,D] row-major, per (h,d). They are different ranks and
 * swapping them is silent. The gate is a VECTOR over the key dimension — each of the 128 key
 * channels of each head forgets at its own data-dependent rate — while beta is ONE SCALAR PER
 * HEAD. Making beta per-channel, or A_log per-channel, both produce finite plausible output.
 *
 * This is factored out of both the decode and the prefill paths rather than fused into either.
 * [vllm] factors it out for decode and fuses it for prefill; factoring it out in BOTH costs one
 * [T,H*D] f32 round-trip and buys an independently testable op — which is the point, because a
 * gate bug is invisible downstream.
 */
__device__ void d_kda_gate(float* __restrict__ g, float* __restrict__ beta,
                           const bf16* __restrict__ g_raw, const bf16* __restrict__ beta_raw,
                           const float* __restrict__ A_log, const float* __restrict__ dt_bias,
                           unsigned T, unsigned H, unsigned D, unsigned mode, float lb,
                           unsigned slice, unsigned nblk) {
    const size_t n = (size_t)T * H * D;
    const size_t chunk = (n + nblk - 1) / nblk;
    const size_t i0 = (size_t)slice * chunk;
    size_t i1 = i0 + chunk;
    if (i1 > n) i1 = n;

    for (size_t i = i0 + threadIdx.x; i < i1; i += PLOW_THREADS) {
        const unsigned hd = (unsigned)(i % ((size_t)H * D)); /* h*D + d */
        const unsigned h = hd / D;
        const float a = __expf(A_log[h]);
        const float s = bf2f(g_raw[i]) + dt_bias[hd];
        g[i] = (mode == PLOW_KDA_GATE_LOWER_BOUND) ? lb * kda_sigmoid(a * s)
                                                   : -a * kda_softplus(s);
    }
    /* beta: T*H elements, its own chunking so it is not stranded on block 0. */
    const size_t nb = (size_t)T * H;
    const size_t cb = (nb + nblk - 1) / nblk;
    const size_t b0 = (size_t)slice * cb;
    size_t b1 = b0 + cb;
    if (b1 > nb) b1 = nb;
    for (size_t i = b0 + threadIdx.x; i < b1; i += PLOW_THREADS)
        beta[i] = kda_sigmoid(bf2f(beta_raw[i]));
}

/* -------------------------------------------------------------------------------------------
 * op 102 — KDA gated delta-rule state update. The core, and a read-modify-write on `state`.
 *
 * THE STATE IS V-FIRST: state[h][v][k], NOT [h][k][v]. K3 passes transpose_state_layout=True
 * (renamed state_v_first upstream). Since V == K == 128 the byte count is identical either way,
 * so a transposed state is garbage WITH EXACTLY THE RIGHT NORM. No magnitude check finds it; the
 * stride arithmetic below IS the assertion.
 *
 * V-first is also what makes the tiling free, via two facts that compose:
 *   1. a v-column (fixed h, fixed v, all K) is 512 CONTIGUOUS bytes;
 *   2. BOTH reductions in the step (S'^T k and S^T q) sum over k for a fixed v, so each output
 *      element is a private, contiguous, 512-byte dot product.
 *
 * LANE MAP. docs/kimi-k3-kda.md §7.2 says "no cross-lane reduction anywhere" AND "4-8 VGPRs/lane";
 * those are inconsistent, because BV*D/512 f32 per lane means 512/BV lanes cooperate on a column
 * and the reduction over k does cross lanes. The resolution keeps every number in that table and
 * drops only the absolute claim:
 *
 *     ONE WAVE OWNS ONE COLUMN. D = 128 = 64 lanes x 2, so a lane holds 2 f32 of state and both
 *     reductions are wave_sum — 6 shuffle steps, no LDS, no __syncthreads, and the whole column is
 *     one wave's private business. Nothing crosses a WAVE, which is the property that matters.
 *
 * Per lane: Sc[2] + q[2] + k[2] + g[2] = 8 f32.
 *
 * SLICE MAP: work item = (head, tile of BV value columns) => H*D/BV items. At H=96, D=128, BV=16
 * that is 768 items over 256 blocks, 3 each — blocks = 256, 100% fill. Blocks stride over items
 * so every block gets the same count. q/k/g for the item's head are staged in LDS once (3*D f32 =
 * 1.5 KiB) and reused by all PLOW_WAVES waves, so the per-head broadcast operands are re-read
 * D/BV times per layer instead of D times.
 *
 * ORDER OF OPERATIONS follows the reference kernel exactly (fla fused_recurrent.py:174-198):
 * decay is applied BEFORE the delta correction is computed, so u is the error against the ALREADY
 * DECAYED state; and o is read off the UPDATED state. The algebraic shortcut
 * o = S'^T q + beta (k.q) u is equivalent but is NOT used — the state is already in registers, so
 * the second pass is free, and matching the reference's association removes a source of fp32
 * divergence rather than trading it for nothing.
 *
 * L2 norm (flags bit 0): eps is INSIDE the sqrt, x / sqrt(sum x^2 + 1e-6), not x / (norm + eps).
 * q is then scaled by `scale` and k is NOT. ||k|| = 1 is load-bearing: it is what makes the delta
 * term an exact rank-1 projector rather than an approximate one.
 *
 * T > 1 runs the same recurrence serially. That is exact at any T — it is the reference's
 * `fused_recurrent` path, which fla uses for q_len == 1 but which is valid for all T — and it is
 * how prefill/decode agreement is tested without a second algorithm.
 */
/* PL = D / PLOW_WAVE, the state elements a lane holds, as a COMPILE-TIME bound. It has to be:
 * `sc[]` is indexed in the inner loop, and a runtime-bounded local array lands in scratch, which
 * is exactly the spill this whole tiling exists to avoid. D=128 => PL=2. */
/* GATE folds op 89 in (op 110). It changes only how `l_g[d]` and `b` are OBTAINED — the slice map,
 * the item map, the LDS layout and every line of the recurrence are shared, which is the point:
 * there is one body, so the fused and unfused paths cannot drift. `g` was an f32 HBM round trip of
 * exactly the expression computed inline here, and an f32 store/load is exact, so the two are
 * BIT-identical rather than merely close. */
template <unsigned PL, bool GATE>
__device__ void d_kda_state_step_t(bf16* __restrict__ o, const bf16* __restrict__ q,
                                   const bf16* __restrict__ k, const bf16* __restrict__ v,
                                   const float* __restrict__ g, const float* __restrict__ beta,
                                   const bf16* __restrict__ g_raw,
                                   const bf16* __restrict__ beta_raw,
                                   const float* __restrict__ a_log,
                                   const float* __restrict__ dt_bias, unsigned gate_mode, float lb,
                                   float* __restrict__ state, unsigned T, unsigned H, unsigned D,
                                   unsigned BV, unsigned flags, float scale, unsigned slice,
                                   unsigned nblk, float* __restrict__ lds, size_t bstride,
                                   const unsigned* __restrict__ parked) {
    const unsigned lane = threadIdx.x & (PLOW_WAVE - 1);
    const unsigned wave = threadIdx.x >> 6;
    const unsigned ntile = D / BV; /* column tiles per head */
    const unsigned items = H * ntile;
    const unsigned cols_per_wave = BV / PLOW_WAVES; /* 2 at BV=16, PLOW_WAVES=8 */

    /* THE ROW AXIS IS PARALLEL WHEN THE ROWS ARE INDEPENDENT SEQUENCES, AND SERIAL OTHERWISE.
     *
     * `bstride != 0` means the T rows are B separate sequences, each with its own carried state at
     * `state + t*bstride`. Nothing in row t's recurrence reads row t-1's, so t is a WORK-ITEM axis
     * exactly like h and tile. `bstride == 0` is a prefill: the T rows are consecutive tokens of
     * ONE sequence threading through ONE state, and there t MUST stay serial.
     *
     * Folding t in is what makes batched decode scale. The item map used to be `H * ntile` at
     * every batch -- 192 items at TP8 with BV=8 -- so a B=16 decode ran 16 rows SERIALLY inside a
     * workgroup count that did not know B existed, on 69 of K3's 93 layers, while 64 of 256 CUs
     * sat idle. The batched-decode design doc asked for "an OUTER slot dimension"; what shipped
     * first was the per-row STRIDE without the axis. */
    const unsigned trep = bstride ? T : 1u;
    const unsigned nitem = items * trep;

    float* l_q = lds;         /* [D] */
    float* l_k = lds + D;     /* [D] */
    float* l_g = lds + 2 * D; /* [D] */

    for (unsigned it = slice; it < nitem; it += nblk) {
        /* Row-major over (row, h, tile), so consecutive slices stay inside one row's state when
         * the row axis is folded in — the same locality the un-folded map had. */
        const unsigned row = bstride ? it / items : 0u;
        const unsigned base = bstride ? it % items : it;
        const unsigned h = base / ntile;
        const unsigned tile = base % ntile;
        float* st_h = state + (size_t)h * D * D;

        /* dt_bias is [H,D] row-major and A_log is PER HEAD — different ranks, and swapping them is
         * silent. Hoisted out of the token loop because neither depends on t. */
        const size_t dtb = (size_t)h * D;
        const float a_h = GATE ? __expf(a_log[h]) : 0.0f;

#if PLOW_KDA_PF_STATE_RESIDENT
        /* TP8 prefill assigns exactly one value column to each wave. Keep that column's two
         * D=128 lane elements live across the serial token recurrence instead of round-tripping
         * them through HBM at every token. Independent decode rows must retain the per-row path. */
        const bool resident = PL == 2 && !bstride && BV == PLOW_WAVES;
        float resident_sc[PL];
        float* resident_col = nullptr;
        if (resident) {
            const unsigned j = tile * BV + wave;
            resident_col = st_h + (size_t)j * D;
#pragma unroll
            for (unsigned r = 0; r < PL; r++)
                resident_sc[r] = resident_col[r * PLOW_WAVE + lane];
        }
#endif

        for (unsigned t = bstride ? row : 0u; t < (bstride ? row + 1u : T); t++) {
            /* PER-ROW PARKED MASK (non-zero = skip). Only a sequence-rows program supplies one,
             * and skipping the
             * row here is the whole point: this recurrence reads AND writes `state[row]` on every
             * dispatch, so a row the server has parked -- a slot in the middle of a chunked
             * prefill, or an idle slot -- would otherwise have its carried state advanced by a
             * garbage token. An append-only KV cache tolerates that (an idle row rewrites a row
             * nothing reads); a recurrence does not. `nullptr` OR an all-zero mask = every row
             * participates, so forgetting to publish one is safe by construction. */
            if (parked && parked[t]) continue;
            const size_t hd = (size_t)t * H * D + (size_t)h * D;
            /* Stage this head's q, k, g once per (item, token) and share across the waves. The L2
             * norm is a whole-head reduction, so it happens here, once, not per column. */
            float qs = 0.0f, ks = 0.0f;
            for (unsigned d = threadIdx.x; d < D; d += PLOW_THREADS) {
                const float qv = bf2f(q[hd + d]), kv = bf2f(k[hd + d]);
                l_q[d] = qv;
                l_k[d] = kv;
                float gv;
                if (GATE) {
                    /* op 89's body, verbatim, evaluated where its only consumer already is. */
                    const float sgm = bf2f(g_raw[hd + d]) + dt_bias[dtb + d];
                    gv = (gate_mode == PLOW_KDA_GATE_LOWER_BOUND) ? lb * kda_sigmoid(a_h * sgm)
                                                                  : -a_h * kda_softplus(sgm);
                } else {
                    gv = g[hd + d];
                }
                l_g[d] = __expf(gv);
                qs += qv * qv;
                ks += kv * kv;
            }
            if (flags & PLOW_KDA_F_QK_L2NORM) {
                /* D <= PLOW_THREADS, so each lane held at most one element; the block reduction is
                 * over whatever the loop above accumulated. eps INSIDE the sqrt. */
                qs = block_sum(qs, lds + 3 * D);
                ks = block_sum(ks, lds + 3 * D + PLOW_WAVES);
                const float rq = scale * rsqrtf(qs + 1e-6f), rk = rsqrtf(ks + 1e-6f);
                for (unsigned d = threadIdx.x; d < D; d += PLOW_THREADS) {
                    l_q[d] *= rq;
                    l_k[d] *= rk;
                }
            } else {
                for (unsigned d = threadIdx.x; d < D; d += PLOW_THREADS) l_q[d] *= scale;
            }
            __syncthreads();

            /* beta is ONE SCALAR PER HEAD, not per channel. Making it per-channel produces finite
             * plausible output, which is why op 89's note says so twice. */
            const float b = GATE ? kda_sigmoid(bf2f(beta_raw[(size_t)t * H + h]))
                                 : beta[(size_t)t * H + h];
            for (unsigned c = 0; c < cols_per_wave; c++) {
                const unsigned j = tile * BV + wave * cols_per_wave + c; /* value column */
                /* PER-ROW STATE. `bstride` is 0 for a PREFILL program, where the T rows are
                 * consecutive tokens of ONE sequence and the recurrence must thread them through
                 * one state — the pointer then does not move and the emitted code is unchanged.
                 * It is `H*D*D` for a BATCHED DECODE program, where the rows are B INDEPENDENT
                 * sequences and sharing a state would run sequence 1's token into sequence 0's.
                 * That distinction is invisible in `T` alone, which is why it is its own
                 * parameter and not inferred (see perf-data/archive/k3/k3-batched-decode-design.md §1). */
                float* col = st_h + (size_t)t * bstride + (size_t)j * D; /* V-FIRST: [v][k] */

                /* decay, in registers: S' = exp(g) * S */
                float sc[PL];
                float pk = 0.0f;
#pragma unroll
                for (unsigned r = 0; r < PL; r++) {
                    const unsigned d = r * PLOW_WAVE + lane;
#if PLOW_KDA_PF_STATE_RESIDENT
                    sc[r] = (resident ? resident_sc[r] : col[d]) * l_g[d];
#else
                    sc[r] = col[d] * l_g[d];
#endif
                    pk += sc[r] * l_k[d];
                }
                pk = wave_sum(pk); /* S'^T k, over k, inside one wave */

                const float u = bf2f(v[hd + j]) - pk;
                const float bu = b * u;

                float pq = 0.0f;
#pragma unroll
                for (unsigned r = 0; r < PL; r++) {
                    const unsigned d = r * PLOW_WAVE + lane;
                    sc[r] += bu * l_k[d]; /* rank-1 write */
#if PLOW_KDA_PF_STATE_RESIDENT
                    if (resident)
                        resident_sc[r] = sc[r];
                    else
                        col[d] = sc[r];
#else
                    col[d] = sc[r];
#endif
                    pq += sc[r] * l_q[d]; /* read the UPDATED state */
                }
                pq = wave_sum(pq);
                if (lane == 0) o[hd + j] = f2bf(pq);
            }
            __syncthreads(); /* l_q/l_k/l_g are rewritten by the next token */
        }
#if PLOW_KDA_PF_STATE_RESIDENT
        if (resident) {
#pragma unroll
            for (unsigned r = 0; r < PL; r++)
                resident_col[r * PLOW_WAVE + lane] = resident_sc[r];
        }
#endif
    }
}

/* D is a runtime immediate, so select the compile-time lane depth here. D=128 (K3) is the only
 * shape any KDA checkpoint uses; the other rungs exist so a wrong D refuses loudly instead of
 * running the D=128 template on the wrong stride. */
#define PLOW_KDA_STEP_RUNGS(GATE_)                                                                \
    if (D == 128)                                                                                 \
        d_kda_state_step_t<2, GATE_>(o, q, k, v, g, beta, g_raw, beta_raw, a_log, dt_bias,        \
                                     gate_mode, lb, state, T, H, D, BV, flags, scale, slice,      \
                                     nblk, lds, bstride, parked);                                                  \
    else if (D == 64)                                                                             \
        d_kda_state_step_t<1, GATE_>(o, q, k, v, g, beta, g_raw, beta_raw, a_log, dt_bias,        \
                                     gate_mode, lb, state, T, H, D, BV, flags, scale, slice,      \
                                     nblk, lds, bstride, parked);                                                  \
    else if (D == 256)                                                                            \
        d_kda_state_step_t<4, GATE_>(o, q, k, v, g, beta, g_raw, beta_raw, a_log, dt_bias,        \
                                     gate_mode, lb, state, T, H, D, BV, flags, scale, slice,      \
                                     nblk, lds, bstride, parked);

__device__ void d_kda_state_step(bf16* __restrict__ o, const bf16* __restrict__ q,
                                 const bf16* __restrict__ k, const bf16* __restrict__ v,
                                 const float* __restrict__ g, const float* __restrict__ beta,
                                 float* __restrict__ state, unsigned T, unsigned H, unsigned D,
                                 unsigned BV, unsigned flags, float scale, unsigned slice,
                                 unsigned nblk, float* __restrict__ lds, size_t bstride,
                                 const unsigned* __restrict__ parked) {
    const bf16 *g_raw = nullptr, *beta_raw = nullptr;
    const float *a_log = nullptr, *dt_bias = nullptr;
    const unsigned gate_mode = 0;
    const float lb = 0.0f;
    PLOW_KDA_STEP_RUNGS(false)
}

/* op 110 — the same recurrence with op 89's gate inlined. Same rungs, same body, same slice map;
 * `g`/`beta` are absent by construction rather than by a null check, because this op has no slot
 * that could name them. */
__device__ void d_kda_state_step_g(bf16* __restrict__ o, const bf16* __restrict__ q,
                                   const bf16* __restrict__ k, const bf16* __restrict__ v,
                                   const bf16* __restrict__ g_raw,
                                   const bf16* __restrict__ beta_raw,
                                   const float* __restrict__ a_log,
                                   const float* __restrict__ dt_bias, unsigned gate_mode, float lb,
                                   float* __restrict__ state, unsigned T, unsigned H, unsigned D,
                                   unsigned BV, unsigned flags, float scale, unsigned slice,
                                   unsigned nblk, float* __restrict__ lds, size_t bstride,
                                   const unsigned* __restrict__ parked) {
    const float *g = nullptr, *beta = nullptr;
    PLOW_KDA_STEP_RUNGS(true)
}

template <bool GATED>
__device__ void d_kda_state_step_packed(
    bf16* o, const bf16* q, const bf16* k, const bf16* v, const float* g,
    const float* beta, const bf16* g_raw, const bf16* beta_raw, const float* a_log,
    const float* dt_bias, unsigned gate_mode, float lb, float* state, unsigned H, unsigned D,
    unsigned BV, unsigned flags, float scale, unsigned slice, unsigned nblk, float* lds,
    const PlowProgram* prog) {
    for (unsigned si = 0; si < prog->n_prefill_spans; ++si) {
        const PlowPrefillSpan* span = plow_packed_prefill_span(prog, si);
        float* ss = state + (size_t)span->state_slot * H * D * D;
        if (span->flags & PLOW_PREFILL_SPAN_RESET_STATE) {
            const unsigned vb = D / BV;
            for (unsigned it = slice; it < H * vb; it += nblk) {
                const unsigned h = it / vb, v0 = (it % vb) * BV;
                for (unsigned d = threadIdx.x; d < D; d += PLOW_THREADS)
                    for (unsigned j = 0; j < BV; ++j)
                        ss[((size_t)h * D + v0 + j) * D + d] = 0.0f;
            }
            __syncthreads();
        }
        const size_t rd = (size_t)span->row0 * H * D;
        const size_t rh = (size_t)span->row0 * H;
        if constexpr (GATED)
            d_kda_state_step_g(o + rd, q + rd, k + rd, v + rd, g_raw + rd,
                               beta_raw + rh, a_log, dt_bias, gate_mode, lb, ss, span->n_rows,
                               H, D, BV, flags & ~PLOW_KDA_F_SEQ_ROWS, scale, slice, nblk, lds,
                               0u, nullptr);
        else
            d_kda_state_step(o + rd, q + rd, k + rd, v + rd,
                             g ? g + rd : nullptr, beta ? beta + rh : nullptr, ss,
                             span->n_rows, H, D, BV, flags & ~PLOW_KDA_F_SEQ_ROWS, scale,
                             slice, nblk, lds, 0u, nullptr);
    }
}
#undef PLOW_KDA_STEP_RUNGS

#if PLOW_KDA_CONV_STEP_DB
__device__ __forceinline__ bf16 kda_conv_db_one(
    const bf16* raw, const float* weight, const float* source, float* target, unsigned channel,
    unsigned W, bool write_state) {
    float win[8], tap[8];
    const unsigned width = W < 8 ? W : 8;
#pragma unroll
    for (unsigned j = 0; j < 8; ++j) {
        win[j] = j < width ? source[(size_t)channel * W + j] : 0.0f;
        tap[j] = j < width ? weight[(size_t)channel * W + j] : 0.0f;
    }
#pragma unroll
    for (unsigned j = 0; j + 1 < 8; ++j) win[j] = win[j + 1];
    win[width - 1] = bf2f(raw[channel]);
    float value = 0.0f;
#pragma unroll
    for (unsigned j = 0; j < 8; ++j) value += win[j] * tap[j];
    if (write_state) {
#pragma unroll
        for (unsigned j = 0; j < 8; ++j)
            if (j < width) target[(size_t)channel * W + j] = win[j];
    }
    return f2bf(act_silu(value));
}

/* Single-row Conv3 + StateStepG candidate. The old and new conv-window banks are distinct for the
 * whole packet, so every value tile can read the old q/k window before tile 0 publishes the next
 * one. This is the cross-workgroup race an in-place fusion cannot avoid.
 *
 * BV must equal the number of waves: every wave owns one value column. D is selected into a
 * compile-time lane depth below, just like d_kda_state_step_t. Heads and the CU count are runtime
 * dimensions; when H*D/BV exceeds nblk a workgroup walks multiple independent value tiles. */
template <int PL>
__device__ __forceinline__ void d_kda_conv_state_step_g_t(
    bf16* output, const bf16* q_raw, const bf16* k_raw, const bf16* v_raw,
    const bf16* gate_raw, const bf16* beta_raw, float* state, const float* wq, const float* wk,
    const float* wv, const float* csq_source, const float* csk_source,
    const float* csv_source, float* csq_target, float* csk_target, float* csv_target,
    const float* a_log, const float* dt_bias, unsigned H, unsigned D, unsigned BV, unsigned W,
    unsigned flags, unsigned gate_mode, float scale, float lb, unsigned slice, unsigned nblk,
    float* lds) {
    if (H == 0 || D != PL * PLOW_WAVE || BV != PLOW_WAVES || W == 0 || W > 8 ||
        !(flags & PLOW_KDA_F_QK_L2NORM))
        __builtin_trap();
    const unsigned items = H * D / BV;
    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 63u;
    const unsigned wave = tid >> 6;
    float* l_q = lds;
    float* l_k = lds + D;
    float* l_g = lds + 2u * D;
    float* l_v = lds + 3u * D + 2u * PLOW_WAVES;
    for (unsigned item = slice; item < items; item += nblk) {
        const unsigned h = item / (D / BV);
        const unsigned tile = item % (D / BV);
        const unsigned hd0 = h * D;
        float qsum = 0.0f, ksum = 0.0f;
        if (tid < D) {
            const unsigned channel = hd0 + tid;
            const bf16 q =
                kda_conv_db_one(q_raw, wq, csq_source, csq_target, channel, W, tile == 0);
            const bf16 k =
                kda_conv_db_one(k_raw, wk, csk_source, csk_target, channel, W, tile == 0);
            l_q[tid] = bf2f(q);
            l_k[tid] = bf2f(k);
            float gate;
            const float gate_input = bf2f(gate_raw[channel]) + dt_bias[channel];
            if (gate_mode == PLOW_KDA_GATE_LOWER_BOUND)
                gate = lb * kda_sigmoid(__expf(a_log[h]) * gate_input);
            else
                gate = -__expf(a_log[h]) * kda_softplus(gate_input);
            l_g[tid] = __expf(gate);
            qsum = l_q[tid] * l_q[tid];
            ksum = l_k[tid] * l_k[tid];
        }
        if (tid < BV) {
            const unsigned channel = hd0 + tile * BV + tid;
            const bf16 v = kda_conv_db_one(v_raw, wv, csv_source, csv_target, channel, W, true);
            l_v[tid] = bf2f(v);
        }
        qsum = block_sum(qsum, lds + 3u * D);
        ksum = block_sum(ksum, lds + 3u * D + PLOW_WAVES);
        const float rq = scale / sqrtf(qsum + 1e-6f);
        const float rk = 1.0f / sqrtf(ksum + 1e-6f);
        if (tid < D) {
            l_q[tid] *= rq;
            l_k[tid] *= rk;
        }
        __syncthreads();

        const unsigned j = tile * BV + wave;
        float* column = state + (size_t)h * D * D + (size_t)j * D;
        float sc[PL];
        float pk = 0.0f;
#pragma unroll
        for (unsigned r = 0; r < PL; ++r) {
            const unsigned d = r * PLOW_WAVE + lane;
            sc[r] = column[d] * l_g[d];
            pk += sc[r] * l_k[d];
        }
        pk = wave_sum(pk);
        const float update = kda_sigmoid(bf2f(beta_raw[h])) * (l_v[wave] - pk);
        float pq = 0.0f;
#pragma unroll
        for (unsigned r = 0; r < PL; ++r) {
            const unsigned d = r * PLOW_WAVE + lane;
            sc[r] += update * l_k[d];
            column[d] = sc[r];
            pq += sc[r] * l_q[d];
        }
        pq = wave_sum(pq);
        if (lane == 0) output[hd0 + j] = f2bf(pq);
        __syncthreads();
    }
}

__device__ void d_kda_conv_state_step_g(
    bf16* output, const bf16* q_raw, const bf16* k_raw, const bf16* v_raw,
    const bf16* gate_raw, const bf16* beta_raw, float* state, const float* wq, const float* wk,
    const float* wv, const float* csq_source, const float* csk_source,
    const float* csv_source, float* csq_target, float* csk_target, float* csv_target,
    const float* a_log, const float* dt_bias, unsigned H, unsigned D, unsigned BV, unsigned W,
    unsigned flags, unsigned gate_mode, float scale, float lb, unsigned slice, unsigned nblk,
    float* lds) {
    if (D == 64)
        d_kda_conv_state_step_g_t<1>(
            output, q_raw, k_raw, v_raw, gate_raw, beta_raw, state, wq, wk, wv, csq_source,
            csk_source, csv_source, csq_target, csk_target, csv_target, a_log, dt_bias, H, D, BV,
            W, flags, gate_mode, scale, lb, slice, nblk, lds);
    else if (D == 128)
        d_kda_conv_state_step_g_t<2>(
            output, q_raw, k_raw, v_raw, gate_raw, beta_raw, state, wq, wk, wv, csq_source,
            csk_source, csv_source, csq_target, csk_target, csv_target, a_log, dt_bias, H, D, BV,
            W, flags, gate_mode, scale, lb, slice, nblk, lds);
    else if (D == 256)
        d_kda_conv_state_step_g_t<4>(
            output, q_raw, k_raw, v_raw, gate_raw, beta_raw, state, wq, wk, wv, csq_source,
            csk_source, csv_source, csq_target, csk_target, csv_target, a_log, dt_bias, H, D, BV,
            W, flags, gate_mode, scale, lb, slice, nblk, lds);
    else
        __builtin_trap();
}
#endif

/* -------------------------------------------------------------------------------------------
 * op 103 — KDA output gate. y[h,d] = RMSNorm_D(o[h,:])[d] * sigmoid(g_raw[h,d]).
 *
 * FusedRMSNormGated(head_dim, eps, activation='sigmoid'). Three things here are easy to get
 * backwards and all three yield plausible-but-wrong output:
 *   - the norm is over D = 128 INSIDE a head, not over H*D = 12288;
 *   - its weight is a single [D] f32 vector SHARED by all H heads (the checkpoint ships exactly
 *     one o_norm.weight per layer, not one per head);
 *   - the sigmoid is applied to the RAW g_proj output and the gate multiplies AFTER the norm, not
 *     before.
 *
 * One wave per (token, head) row: T*H items, the reduction is a wave_sum over D/64 elements per
 * lane, and nothing crosses a wave. The packet therefore needs ceil(T*H/PLOW_WAVES) workgroups:
 * 12 at TP1 B1 and 2 at TP8 B1. Folding this into op 102's epilogue instead needs a grid-wide
 * barrier because a head's D outputs are spread over D/BV workgroups there.
 */
__device__ void d_kda_gated_norm(bf16* __restrict__ y, const bf16* __restrict__ o,
                                 const float* __restrict__ norm_w, const bf16* __restrict__ g_raw,
                                 unsigned T, unsigned H, unsigned D, float eps, unsigned slice,
                                 unsigned nblk) {
    const unsigned lane = threadIdx.x & (PLOW_WAVE - 1);
    const unsigned wave = threadIdx.x >> 6;
    const unsigned rows = T * H;
    for (unsigned r = slice * PLOW_WAVES + wave; r < rows; r += nblk * PLOW_WAVES) {
        const size_t base = (size_t)r * D;
        float ss = 0.0f;
        for (unsigned d = lane; d < D; d += PLOW_WAVE) {
            const float x = bf2f(o[base + d]);
            ss += x * x;
        }
        const float inv = rsqrtf(wave_sum(ss) / (float)D + eps);
        for (unsigned d = lane; d < D; d += PLOW_WAVE)
            y[base + d] = f2bf(bf2f(o[base + d]) * inv * norm_w[d] * kda_sigmoid(bf2f(g_raw[base + d])));
    }
}

#endif /* PLOW_OP_KDA_H */
