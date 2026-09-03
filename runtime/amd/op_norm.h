/* op_norm.h — RMSNorm family (CDNA).
 *
 * Three call sites in a transformer block, one implementation each:
 *   d_rmsnorm       — norm over the hidden dim, with weight
 *   d_headnorm_rope — norm over head_dim, then optional RoPE (q_norm/k_norm/v_norm)
 *   d_rowrms        — the RMS scalars ONLY, so a GEMM can apply the norm in its
 *                     prologue instead of round-tripping a normed activation
 *                     through HBM (see op_gemm.h, d_gemm_norm)
 *
 * NUMERICS (Gemma 4; HF modeling_gemma4.py):
 *   y = x * pow(mean(x^2) + eps, -0.5) * w        <- eps INSIDE the power
 * The weight is a PLAIN multiply. Gemma 1/2/3 used (1 + w) with zero-init
 * weights; Gemma 4 switched to ones-init and dropped the +1. Adding it back
 * scales every activation by ~2 and still produces fluent text.
 *
 * `gamma == nullptr` is the weightless variant (`with_scale=False`), which Gemma
 * applies to V on EVERY layer as `v_norm`. It has no checkpoint tensor, so it is
 * the easiest part of the model to omit entirely.
 */
#ifndef PLOW_OP_NORM_H
#define PLOW_OP_NORM_H

#include "amd_common.h"
#include "packed_prefill.h"

/* RN_REG / RN_VEC moved to amd_common.h: op_gemm.h's fused-norm GEMV (`norm == 2`,
 * `gemv_norm_lds`) must reduce with the SAME per-thread element map as `d_rmsnorm` below to
 * stay bit-exact, and it is included BEFORE this header. A constant two headers must agree on
 * is one constant, not two — this pair being copied is exactly how a fused arm ends up
 * "mathematically equivalent" and numerically different. */

/* RMSNorm over `feat`. One workgroup per row, strided by nblk.
 *
 * The row is held in REGISTERS across the reduction, and this is not a micro-optimisation:
 * a decode step norms a SINGLE row, so exactly one CU has work while the other 255 wait on
 * the counter. The streaming version made two HBM passes (sum, then normalise) with a
 * non-unrolled loop, so it was pure load latency -- 52 us for 5376 elements, x241 packets =
 * 12.5 ms of a 45 ms token, with the whole machine stalled behind it.
 *
 * PRODUCER, THEN CONSUMER. Every operand -- the input tile AND the weight -- is staged on-chip
 * in ONE burst before any arithmetic happens; the reduction and the scaling then run entirely
 * out of registers, and HBM is not touched again until the writeback.
 *
 * That ordering is the whole point. This op used to load x, reduce it, and only THEN go and
 * fetch gamma -- a SECOND, fully serialised HBM round trip, sitting behind a block-wide barrier,
 * on the critical path of an op that 255 other CUs are stalled on. d_norm_residual was worse
 * still: it staged `b` up front but fetched BOTH `a` and gamma after the barrier.
 *
 * A decode norm is one CU, one row, ~21 KB. Issued together, that is ONE ~1.3 us round trip and
 * nothing more; issued in two dependent phases it is two. Nothing else in the op costs anything
 * like as much, so the load ORDER was most of its runtime. */
/* Workgroup max, same shape (and same `part` reuse contract) as block_sum. */
__device__ __forceinline__ float block_max(float v, float* part) {
#pragma unroll
    for (int off = 32; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor(v, off, PLOW_WAVE));
    const unsigned wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    if (lane == 0) part[wave] = v;
    __syncthreads();
    float t = part[0];
#pragma unroll
    for (int i = 1; i < PLOW_WAVES; i++) t = fmaxf(t, part[i]);
    __syncthreads();
    return t;
}

__device__ void d_rmsnorm(bf16* __restrict__ out, const bf16* __restrict__ x,
                          const bf16* __restrict__ gamma, unsigned rows, unsigned feat,
                          float eps, unsigned out_row0, unsigned slice, unsigned nblk, float* part,
                          unsigned char* __restrict__ xq = nullptr,
                          float* __restrict__ ascale = nullptr
#if PLOW_PACKED_PREFILL_CONSUMERS
                          ,
                          const PlowProgram* packed = nullptr,
                          unsigned packed_slot_stride = 0
#endif
                          ) {
    /* feat % 8 == 0 is what lets the row be read 16 bytes at a time; Gemma's 5376 is. */
    const bool fits = (feat <= RN_REG * PLOW_THREADS) && ((feat & 7u) == 0);
    const auto* xg = as_glob(x);
    const auto* gg = as_glob(gamma);
    auto* og = as_glob(out);
    for (unsigned row = slice; row < rows; row += nblk) {
#if PLOW_PACKED_PREFILL_CONSUMERS
        const PlowPackedRow prow = plow_packed_prefill_row(packed, row);
        if (!prow.active) continue;
#endif
        const size_t base = (size_t)row * feat;
        /* out_row0 offsets the OUTPUT row only (input stays at `base`): GLM's decode step norms the
         * current token (row 0 of x) into the latent KV cache at row = out_row0 (the sequence pos),
         * patched per step. Default 0 => in-place, every existing RMSNORM bit-identical. */
#if PLOW_PACKED_PREFILL_CONSUMERS
        const size_t out_row = packed_slot_stride
                                   ? plow_packed_prefill_cache_row(prow, packed_slot_stride,
                                                                  out_row0 + row)
                                   : out_row0 + row;
#else
        const size_t out_row = out_row0 + row;
#endif
        const size_t obase = out_row * feat;
        if (fits) {
            /* PRODUCE: the row AND its weight, in one burst. Both are issued before anything
             * is waited on, so they cost ONE round trip between them, not one each. */
            bf16v8 v[RN_VEC], w[RN_VEC];
#pragma unroll
            for (int c = 0; c < RN_VEC; c++) {
                const unsigned i = (threadIdx.x + (unsigned)c * PLOW_THREADS) * 8;
                v[c] = bf16v8_zero();
                w[c] = bf16v8_zero();
                if (i < feat) {
                    v[c] = ld_glob8(xg + base + i);
                    if (gamma) w[c] = ld_glob8(gg + i);
                }
            }
            /* CONSUME: reduce, scale, store. No HBM read from here on. */
            float ss = 0.0f;
#pragma unroll
            for (int c = 0; c < RN_VEC; c++)
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    const float f = bf2f(v[c][j]);
                    ss += f * f;
                }
            const float inv = rsqrtf(block_sum(ss, part) / (float)feat + eps);
#pragma unroll
            for (int c = 0; c < RN_VEC; c++) {
                const unsigned i = (threadIdx.x + (unsigned)c * PLOW_THREADS) * 8;
                if (i < feat) {
                    bf16v8 o;
#pragma unroll
                    for (int j = 0; j < 8; j++) {
                        const float g = gamma ? bf2f(w[c][j]) : 1.0f;
                        o[j] = f2bf(bf2f(v[c][j]) * inv * g);
                    }
                    st_glob8(og + obase + i, o);
                    v[c] = o; /* keep the ROUNDED row for the fused quant below */
                }
            }
            /* FUSED w8a8 ACTIVATION QUANT (t[3]/t[4] set). Quantizes the just-ROUNDED bf16
             * outputs exactly as d_quant_fp8 would after re-reading them: fmax is order-
             * independent, the scale formula and the encoder are the same code, so xq/ascale
             * are BIT-IDENTICAL to the separate QUANT_FP8 packet this replaces — what it
             * deletes is that packet's full read+write pass over the row and its serial
             * gate. Only emitted with out_row0 == 0, so xq rows index like the packet's. */
            if (xq) {
                float amax = 0.0f;
#pragma unroll
                for (int c = 0; c < RN_VEC; c++) {
                    const unsigned i = (threadIdx.x + (unsigned)c * PLOW_THREADS) * 8;
                    if (i < feat)
#pragma unroll
                        for (int j = 0; j < 8; j++) amax = fmaxf(amax, fabsf(bf2f(v[c][j])));
                }
                amax = block_max(amax, part);
                const float as = fmaxf(amax * (1.0f / 448.0f), 1e-12f);
                const float qinv = 1.0f / as;
                auto* const xqg = as_glob(xq);
                if (threadIdx.x == 0) st_act<float>(&as_glob(ascale)[row], as);
#pragma unroll
                for (int c = 0; c < RN_VEC; c++) {
                    const unsigned i = (threadIdx.x + (unsigned)c * PLOW_THREADS) * 8;
                    if (i < feat) {
#pragma unroll
                        for (int j = 0; j < 8; j += 2) {
                            const float qa = bf2f(v[c][j]) * qinv;
                            const float qb = bf2f(v[c][j + 1]) * qinv;
#if PLOW_CDNA4
                            const unsigned pk = __builtin_amdgcn_cvt_pk_fp8_f32(qa, qb, 0u, false);
#else
                            const unsigned pk = (unsigned)plow_f32_to_fp8_ocp(qa) |
                                                ((unsigned)plow_f32_to_fp8_ocp(qb) << 8);
#endif
                            st_act1_u8(&xqg[base + i + j], (unsigned char)(pk & 0xffu));
                            st_act1_u8(&xqg[base + i + j + 1], (unsigned char)((pk >> 8) & 0xffu));
                        }
                    }
                }
            }
        } else {
            float ss = 0.0f;
            for (unsigned i = threadIdx.x; i < feat; i += PLOW_THREADS) {
                const float v = bf2f(x[base + i]);
                ss += v * v;
            }
            const float inv = rsqrtf(block_sum(ss, part) / (float)feat + eps);
            float amax = 0.0f;
            for (unsigned i = threadIdx.x; i < feat; i += PLOW_THREADS) {
                const float g = gamma ? bf2f(gamma[i]) : 1.0f;
                const bf16 o = f2bf(bf2f(x[base + i]) * inv * g);
                st_act1(&out[obase + i], o);
                amax = fmaxf(amax, fabsf(bf2f(o)));
            }
            /* Fused quant, non-resident arm: each thread encodes the elements it just
             * produced (the encoder is ELEMENTWISE — pairing in d_quant_fp8 is only an
             * encode-width optimization — so the bytes are identical), avoiding any
             * cross-thread re-read of the just-written row. */
            if (xq) {
                amax = block_max(amax, part);
                const float as = fmaxf(amax * (1.0f / 448.0f), 1e-12f);
                const float qinv = 1.0f / as;
                auto* const xqg = as_glob(xq);
                if (threadIdx.x == 0) st_act<float>(&as_glob(ascale)[row], as);
                for (unsigned i = threadIdx.x; i < feat; i += PLOW_THREADS) {
                    const float g = gamma ? bf2f(gamma[i]) : 1.0f;
                    const float qa = bf2f(f2bf(bf2f(x[base + i]) * inv * g)) * qinv;
#if PLOW_CDNA4
                    const unsigned pk = __builtin_amdgcn_cvt_pk_fp8_f32(qa, 0.0f, 0u, false);
#else
                    const unsigned pk = (unsigned)plow_f32_to_fp8_ocp(qa);
#endif
                    st_act1_u8(&xqg[base + i], (unsigned char)(pk & 0xffu));
                }
            }
        }
    }
}

/* Row RMS scalars only: rms[row] = pow(mean(x^2) + eps, -0.5), in f32.
 * Fused-norm GEMM consumes these, so the normalized activation never touches
 * HBM. Cost is one extra pass over x (M*K reads) but it saves a full M*K write
 * plus a full M*K read. */
__device__ void d_rowrms(float* __restrict__ rms, const bf16* __restrict__ x, unsigned rows,
                         unsigned feat, float eps, unsigned slice, unsigned nblk,
                         float* part) {
    const auto* xg = as_glob(x);
    for (unsigned row = slice; row < rows; row += nblk) {
        const size_t base = (size_t)row * feat;
        float ss = 0.0f;
        for (unsigned i = threadIdx.x; i < feat; i += PLOW_THREADS) {
            const float v = bf2f(xg[base + i]);
            ss += v * v;
        }
        const float inv = rsqrtf(block_sum(ss, part) / (float)feat + eps);
        if (threadIdx.x == 0) st_act<float>(&rms[row], inv);
    }
}

/* LayerNorm WITH bias (mean-subtract) over `feat`, one workgroup per row.        [GLM52-DSA]
 *   y = (x - mean(x)) * rsqrt(var(x) + eps) * gamma + beta   (var = mean(x^2) - mean(x)^2)
 * This is the DSA lightning-indexer key-norm (GLM-5.2 `indexer.k_norm`, an nn.LayerNorm(index_head_dim
 * =128, eps=1e-6) WITH bias) — the only LayerNorm in the model; every other norm is RMS. Small row
 * (feat=128) so the streaming path is fine (one CU, ~256 B). Two reductions (sum, sumsq) in one pass;
 * gamma/beta are the learned weight/bias. `out_row0` offsets only the output row (mirrors d_rmsnorm),
 * so the decode step can write the current token's index-key into its [ctx][DI] cache slot. */
__device__ void d_layernorm_bias(bf16* __restrict__ out, const bf16* __restrict__ x,
                                 const bf16* __restrict__ gamma, const bf16* __restrict__ beta,
                                 unsigned rows, unsigned feat, float eps, unsigned out_row0,
                                 unsigned slice, unsigned nblk, float* part) {
    const auto* xg = as_glob(x);
    const auto* gg = as_glob(gamma);
    const auto* bg = as_glob(beta);
    auto* og = as_glob(out);
    for (unsigned row = slice; row < rows; row += nblk) {
        const size_t base = (size_t)row * feat;
        const size_t obase = (size_t)(out_row0 + row) * feat;
        float s = 0.0f, ss = 0.0f;
        for (unsigned i = threadIdx.x; i < feat; i += PLOW_THREADS) {
            const float v = bf2f(xg[base + i]);
            s += v;
            ss += v * v;
        }
        const float mean = block_sum(s, part) / (float)feat;
        const float msq = block_sum(ss, part) / (float)feat;
        const float inv = rsqrtf(msq - mean * mean + eps);
        for (unsigned i = threadIdx.x; i < feat; i += PLOW_THREADS) {
            const float g = gamma ? bf2f(gg[i]) : 1.0f;
            const float b = beta ? bf2f(bg[i]) : 0.0f;
            st_act1(&og[obase + i], f2bf((bf2f(xg[base + i]) - mean) * inv * g + b));
        }
    }
}

/* Per-head RMSNorm, then optional RoPE. x is [ntok, nhead, hd]; norm is over hd.
 * One wave per (token, head).
 *
 * hd is 256 or 512 — always a multiple of 64 — so lane `l` owns elements
 * {l, l+64, ...} and every RoPE pair (i, i + hd/2) lands in the SAME lane. The
 * rotation therefore needs zero cross-lane traffic.
 *
 * RoPE is half-split (pairs i with i+hd/2), NOT interleaved. On the global
 * layers head_dim is 512 and only the first 64 frequencies are non-zero, so the
 * pairs that actually rotate are (i, i+256) for i < 64 — the partner is i+256,
 * not i+64. None of that is branched on here: "proportional" RoPE zero-pads
 * inv_freq, so the NoPE dims arrive as cos=1/sin=0 and pass through bit-exact.
 * One kernel serves both layer geometries.
 *
 * `out_row0` lets the result land directly at a row offset of a KV-cache tensor,
 * so writing K/V into the cache is not a separate copy. */
/* HD is a TEMPLATE parameter, and that is not cosmetic.
 *
 * `E = hd >> 6` -- the elements each lane owns -- used to be a RUNTIME value, so
 * `for (e = 0; e < E; e++) { v[e] = x[...]; ss += v[e]*v[e]; }` could not be unrolled. Each
 * iteration's load was therefore issued, waited on, and consumed before the next one was even
 * addressed: E DEPENDENT HBM ROUND TRIPS instead of one. That is why this op cost 9.3 us to
 * move 16 KB on 4 CUs while 252 others stalled behind it, and why it was the second-biggest
 * item on the decode critical path.
 *
 * Gemma has exactly two head dims, so the switch is closed -- the same argument the attention
 * ops already made for their O accumulator. */
/* THE WORK ITEM IS A HEAD AND A HEAD IS ONE WAVE, and that caps the op's parallelism.
 *
 * `slice * PLOW_WAVES + wave_in_blk` packs 8 heads into every workgroup, so decode's 32 q heads
 * land on ceil(32/8) = 4 CUs. Handing the same 32 waves to 32 workgroups instead was tried
 * (n=6 interleaved, Gemma-4-31B bf16, ctx 1024): 18.057 -> 18.460 ms/token, a REGRESSION. Same
 * wave count, 8x the L1s, and it still lost — the op is not limited by one CU's issue bandwidth,
 * it is limited by the DEPENDENT round trip below, and the extra 28 workgroups per packet only
 * bought 5040 more workgroup-packets of gate participation per token. `hd=256` is exactly one
 * wave's 64 lanes x 4 elements and the lane->element map is what keeps each RoPE pair
 * (i, i+hd/2) inside one lane, so a head cannot be split across waves without cross-lane
 * traffic. Do not widen this without changing that map first. */
/* INTERLEAVE (GPT-J / GLM-5.2 style, template flag, default OFF): rotate ADJACENT even/odd pairs
 * (x[2i], x[2i+1]) instead of the half-split (i, i+H2). GLM-5.2 MLA applies interleaved partial RoPE
 * to the 64-dim rope slice of q (per head) and the shared k_rope (rope_theta=8e6). With the lane-
 * strided layout (lane l owns {l, l+64, ...}) an element's interleaved partner (index XOR 1) lives in
 * the ADJACENT lane (l XOR 1) at the SAME e (since e*64 is even), so ONE __shfl_xor(.,1,64) fetches it
 * — the only cross-lane traffic RoPE needs, and only on this branch. cos/sin use the SAME H2-per-
 * position table as the half-split path (freq index = element_index >> 1). Default false keeps every
 * existing (half-split) instantiation bit-identical. */
/* THE DEPENDENT `pos` LOAD IS REAL AND IT COSTS NOTHING. MEASURED, so do not re-propose it.
 *
 * `p = pg[t] * H2` below is a load whose RESULT is the address of the cos/sin loads, so the
 * compiler has no choice but to emit
 *      global_load_dword v30, v[30:31], off   (pos[t])
 *      s_waitcnt vmcnt(0)                     <- full drain
 *      global_load_dword v36, v[34:35], off   (cos, addressed FROM the pos result)
 * -- that is the actual gfx950 disassembly. Two serial round trips where one would do, 180 times
 * per decode token, with nothing else ready behind the op. It looks like an obvious win.
 *
 * It was built: `pos_imm`, an extra instruction field carrying (position of row 0)+1 so that 0
 * kept the tensor load, patched per step by the host at every headnorm site the way it already
 * patches `out_row0` at the k/v ones. Token-identical. The ISA confirmed the load and its drain
 * moved under a branch the fast path skips. Gemma-4-31B bf16, ctx 1024, gfx950, interleaved,
 * paired per round, n=14:
 *
 *     median +0.003 ms/token, mean +0.066, sd 0.196  => the win is bounded above by ~0.04 ms
 *
 * A NULL, against the ~0.23 ms predicted from "two 1.3 us round trips x 180 packets". The premise
 * is what is wrong: `pos` is FOUR BYTES read 180 times per token, so it is permanently L1/L2 hot
 * and the "round trip" is a cache hit of order 100 ns, not 1.3 us -- and with 8 waves per
 * workgroup even that is overlapped. Reverted.
 *
 * Widening the op was measured too and it REGRESSES; see the comment above the loop below. So
 * `headnorm_rope`'s 0.385-0.80 ms/token is not recoverable by either route. */
template <int HD, bool INTERLEAVE = false>
__device__ void d_headnorm_rope(bf16* __restrict__ out, const bf16* __restrict__ x,
                                const bf16* __restrict__ gamma,
                                const float* __restrict__ cosb, const float* __restrict__ sinb,
                                const int* __restrict__ pos, unsigned ntok, unsigned nhead,
                                float eps, unsigned out_row0, unsigned out_stride, unsigned kv_mask,
                                unsigned skip_norm, unsigned slice,
                                unsigned nblk, unsigned n_batch_kv = 0
#if PLOW_PACKED_PREFILL_CONSUMERS
                                ,
                                const PlowProgram* packed = nullptr,
                                unsigned packed_slot_stride = 0
#endif
                                ) {
    constexpr unsigned hd = HD;
    const unsigned lane = threadIdx.x & 63;
    const unsigned wave_in_blk = threadIdx.x >> 6; /* PLOW_WAVES per workgroup */
    const unsigned total = ntok * nhead;
    constexpr unsigned E = HD >> 6; /* elements per lane: 4 (hd=256) or 8 (hd=512) */

    /* The lane-strided access stays -- it is what keeps each RoPE pair (i, i + hd/2) inside
     * ONE lane, so the rotation needs no cross-lane traffic. It cannot be widened to 16-byte
     * vectors without breaking that. But it can at least be a global_load rather than a flat
     * one: 64 lanes x 2 B is still a coalesced 128-byte request. */
    const auto* xg = as_glob(x);
    const auto* gg = as_glob(gamma);
    const auto* cg = as_glob(cosb);
    const auto* sg = as_glob(sinb);
    const auto* pg = as_glob(pos);
    auto* og = as_glob(out);

    for (unsigned w = slice * PLOW_WAVES + wave_in_blk; w < total; w += nblk * PLOW_WAVES) {
        const unsigned t = w / nhead, hh = w % nhead;
#if PLOW_PACKED_PREFILL_CONSUMERS
        const PlowPackedRow prow = plow_packed_prefill_row(packed, t);
        if (!prow.active) continue;
        const unsigned position =
            plow_packed_prefill_position(prow, pg ? (unsigned)pg[t] : out_row0 + t);
#else
        const unsigned position = pg ? (unsigned)pg[t] : out_row0 + t;
#endif
        const size_t ibase = ((size_t)t * nhead + hh) * hd;
        /* out_stride != 0 selects the HEAD-MAJOR KV-cache layout, [kv_head][ctx][hd]: this
         * wave's 512 bytes land inside its own head's contiguous block, so flash_decode can
         * stream a head end-to-end instead of striding n_kv_head*hd between every row. See
         * dev_isa.h. out_stride == 0 is the plain [ntok][nhead][hd] output (the q norm). */
        /* `& kv_mask` is the sliding-window RING (dev_isa.h). Full layers pass 0xFFFFFFFF, so
         * this is one v_and_b32 that does nothing — no branch, no runtime flag. */
        /* BATCH>1 decode (i6 = n_batch_kv, mirrors runtime/nvidia/op_norm.cuh): token t IS
         * sequence t and writes into ITS OWN batch-major ring at ITS OWN position pos[t]:
         *   obase = ((t*nhead + hh)*out_stride + (pos[t] & kv_mask)) * hd.
         * Without this the batch index is consumed as a TIME offset (out_row0 + t) into a
         * single shared ring, so sequences 1..B-1 write K/V over each other's rows — and a
         * "sequence 0 matches B=1" check cannot see it, because sequence 0 is the one row the
         * legacy formula gets right. n_batch_kv == 0 keeps the legacy path byte-identical, so
         * every prefill packet and B=1 decode are unchanged. */
        const size_t obase =
#if PLOW_PACKED_PREFILL_CONSUMERS
            packed_slot_stride && prow.span
                ? (((size_t)prow.span->slot * nhead + hh) * packed_slot_stride +
                   (position & kv_mask)) * hd
                :
#endif
                  out_stride
                ? (n_batch_kv != 0
                       ? ((size_t)(t * nhead + hh) * out_stride + (position & kv_mask)) * hd
                       : ((size_t)hh * out_stride + ((out_row0 + t) & kv_mask)) * hd)
                : ((size_t)(out_row0 + t) * nhead + hh) * hd;

        /* PRODUCE: the head AND its weight, all E of each, issued before anything is waited
         * on. One round trip, not E of them. */
        float v[E], g[E];
#pragma unroll
        for (unsigned e = 0; e < E; e++) {
            v[e] = bf2f(xg[ibase + lane + e * 64]);
            g[e] = gamma ? bf2f(gg[lane + e * 64]) : 1.0f;
        }
        /* skip_norm: Llama has NO q/k norm and NEITHER model norms V — the head passes through
         * to RoPE (q/k) or straight to the cache (v) unnormalized. Gemma leaves skip_norm 0, so
         * its weightless v_norm and learned q/k norms are unchanged. gamma is still applied when
         * present (it is NONE for every skip_norm path today, so g[e]==1). */
        float inv = 1.0f;
        if (!skip_norm) {
            float ss = 0.0f;
#pragma unroll
            for (unsigned e = 0; e < E; e++) ss += v[e] * v[e];
            inv = rsqrtf(wave_sum(ss) / (float)hd + eps);
        }
#pragma unroll
        for (unsigned e = 0; e < E; e++) v[e] = v[e] * inv * g[e];

        if (cosb) {
            constexpr unsigned H2 = HD >> 1;
            const size_t p = (size_t)position * H2;
            float r[E];
            if constexpr (!INTERLEAVE) {
                constexpr unsigned EH = H2 >> 6; /* lane-local stride to the half-split partner */
#pragma unroll
                for (unsigned e = 0; e < E; e++) {
                    const unsigned i = lane + e * 64;
                    const unsigned j = (i < H2) ? i : (i - H2);
                    const float c = cg[p + j], s = sg[p + j];
                    r[e] = (e < EH) ? (v[e] * c - v[e + EH] * s)   /* i in [0, H2)  */
                                    : (v[e] * c + v[e - EH] * s);  /* i in [H2, hd) */
                }
            } else {
                /* interleaved: pair (2m, 2m+1). partner index = i XOR 1 -> lane (lane XOR 1), same e.
                 * even i: r =  v*cos - partner*sin ; odd i: r = v*cos + partner*sin. */
#pragma unroll
                for (unsigned e = 0; e < E; e++) {
                    const unsigned i = lane + e * 64;
                    const float c = cg[p + (i >> 1)], s = sg[p + (i >> 1)];
                    const float partner = __shfl_xor(v[e], 1, PLOW_WAVE);
                    r[e] = ((i & 1u) == 0u) ? (v[e] * c - partner * s)
                                            : (v[e] * c + partner * s);
                }
            }
            for (unsigned e = 0; e < E; e++) v[e] = r[e];
        }

        for (unsigned e = 0; e < E; e++) st_act1(&og[obase + lane + e * 64], f2bf(v[e]));
    }
}

/* FP8 (e4m3) KV-cache twin of d_headnorm_rope. Identical math — the norm/RoPE produce the SAME
 * v[e] — but the store quantizes to e4m3 with a PER-ROW scale so flash-decode reads half the bytes.
 *
 * A "row" is one (token, kv_head) K/V vector of `hd` elements, held across the 64 lanes of this
 * wave (E per lane). Its amax over the whole row picks the scale: scale = amax/448 maps the largest
 * element onto e4m3's full range, and the stored byte is round(v/scale). flash-decode multiplies the
 * decoded value back by `scale` (per row), so the only loss is e4m3's 3-mantissa-bit quantization —
 * which attention tolerates (CK/AITER both ship fp8 KV). out is the uint8 cache; `scale` is the
 * f32[kv_head][ctx] scale array, same head-major layout and RING as the cache. out_stride != 0
 * always here (this path is k/v only). */
template <int HD>
__device__ void d_headnorm_rope_fp8(unsigned char* __restrict__ out, float* __restrict__ scale,
                                    const bf16* __restrict__ x, const bf16* __restrict__ gamma,
                                    const float* __restrict__ cosb, const float* __restrict__ sinb,
                                    const int* __restrict__ pos, unsigned ntok, unsigned nhead,
                                    float eps, unsigned out_row0, unsigned out_stride,
                                    unsigned kv_mask, unsigned skip_norm, unsigned slice,
                                    unsigned nblk, unsigned n_batch_kv = 0
#if PLOW_PACKED_PREFILL_CONSUMERS
                                    ,
                                    const PlowProgram* packed = nullptr,
                                    unsigned packed_slot_stride = 0
#endif
                                    ) {
    constexpr unsigned hd = HD;
    const unsigned lane = threadIdx.x & 63;
    const unsigned wave_in_blk = threadIdx.x >> 6;
    const unsigned total = ntok * nhead;
    constexpr unsigned E = HD >> 6;

    const auto* xg = as_glob(x);
    const auto* gg = as_glob(gamma);
    const auto* cg = as_glob(cosb);
    const auto* sg = as_glob(sinb);
    const auto* pg = as_glob(pos);
    auto* og = as_glob(out);
    auto* scg = as_glob(scale);

    for (unsigned w = slice * PLOW_WAVES + wave_in_blk; w < total; w += nblk * PLOW_WAVES) {
        const unsigned t = w / nhead, hh = w % nhead;
#if PLOW_PACKED_PREFILL_CONSUMERS
        const PlowPackedRow prow = plow_packed_prefill_row(packed, t);
        if (!prow.active) continue;
        const unsigned position =
            plow_packed_prefill_position(prow, pg ? (unsigned)pg[t] : out_row0 + t);
#else
        const unsigned position = pg ? (unsigned)pg[t] : out_row0 + t;
#endif
        const size_t ibase = ((size_t)t * nhead + hh) * hd;
        /* KV row index. n_batch_kv != 0 = BATCH>1 decode: sequence t's own batch-major ring at
         * its own pos[t] (see the bf16 twin above). The per-row `scale` array shares this row,
         * so both follow the same formula. */
        const size_t row =
#if PLOW_PACKED_PREFILL_CONSUMERS
                           packed_slot_stride && prow.span
                               ? ((size_t)prow.span->slot * nhead + hh) * packed_slot_stride +
                                     (position & kv_mask)
                               :
#endif
                                 n_batch_kv != 0
                               ? (size_t)(t * nhead + hh) * out_stride + (position & kv_mask)
                               : (size_t)hh * out_stride + ((out_row0 + t) & kv_mask);
        const size_t obase = row * hd;

        float v[E], g[E];
#pragma unroll
        for (unsigned e = 0; e < E; e++) {
            v[e] = bf2f(xg[ibase + lane + e * 64]);
            g[e] = gamma ? bf2f(gg[lane + e * 64]) : 1.0f;
        }
        float inv = 1.0f;
        if (!skip_norm) {
            float ss = 0.0f;
#pragma unroll
            for (unsigned e = 0; e < E; e++) ss += v[e] * v[e];
            inv = rsqrtf(wave_sum(ss) / (float)hd + eps);
        }
#pragma unroll
        for (unsigned e = 0; e < E; e++) v[e] = v[e] * inv * g[e];

        if (cosb) {
            constexpr unsigned H2 = HD >> 1;
            constexpr unsigned EH = H2 >> 6;
            const size_t p = (size_t)position * H2;
            float r[E];
#pragma unroll
            for (unsigned e = 0; e < E; e++) {
                const unsigned i = lane + e * 64;
                const unsigned j = (i < H2) ? i : (i - H2);
                const float c = cg[p + j], s = sg[p + j];
                r[e] = (e < EH) ? (v[e] * c - v[e + EH] * s) : (v[e] * c + v[e - EH] * s);
            }
            for (unsigned e = 0; e < E; e++) v[e] = r[e];
        }

        /* PER-ROW SCALE: amax over the whole hd-vector (this lane's E elements, then across the
         * wave), map to e4m3's 448. inv==0 when the row is all zeros, so v*inv is 0 and quant_fp8
         * gives 0 — no NaN from 448/0. */
        float amax = 0.0f;
#pragma unroll
        for (unsigned e = 0; e < E; e++) amax = fmaxf(amax, fabsf(v[e]));
        amax = wave_max(amax);
        const float qinv = (amax > 0.0f) ? (PLOW_FP8_E4M3_MAX / amax) : 0.0f;
        if (lane == 0) st_act<float>(&scg[row], amax * (1.0f / PLOW_FP8_E4M3_MAX));
#pragma unroll
        for (unsigned e = 0; e < E; e++) st_act1_u8(&og[obase + lane + e * 64], quant_fp8(v[e] * qinv));
    }
}

/* out = (a + RMSNorm(b, gamma)) * scale — Gemma's SANDWICH tail, in one packet.
 *
 * The reference is
 *     hidden = residual + post_norm(sublayer_out)     (and, on the 2nd one, * layer_scalar)
 * which we were emitting as an RMSNORM packet followed by a RESIDUAL packet: two global
 * gates, and the norm alone is a row reduction, so in decode ONE workgroup does it while 255
 * wait. Fusing halves both. The row is held in registers exactly as in d_rmsnorm, so `b` is
 * read once, not twice. */
__device__ void d_norm_residual(bf16* __restrict__ out, const bf16* __restrict__ a,
                                const bf16* __restrict__ b, const bf16* __restrict__ gamma,
                                unsigned rows, unsigned feat, float eps, float scale,
                                unsigned slice, unsigned nblk, float* part) {
    const bool fits = (feat <= RN_REG * PLOW_THREADS) && ((feat & 7u) == 0);
    const auto* ag = as_glob(a);
    const auto* bg = as_glob(b);
    const auto* gg = as_glob(gamma);
    auto* og = as_glob(out);
    for (unsigned row = slice; row < rows; row += nblk) {
        const size_t base = (size_t)row * feat;
        if (fits) {
            /* PRODUCE: all THREE operands -- the sublayer output b, the residual a, and the
             * weight -- issued together. This op used to fetch a and gamma only AFTER the
             * barrier, which is why it cost 3.2 us against rmsnorm's 2.8: two round trips. */
            bf16v8 v[RN_VEC], av[RN_VEC], w[RN_VEC];
#pragma unroll
            for (int c = 0; c < RN_VEC; c++) {
                const unsigned i = (threadIdx.x + (unsigned)c * PLOW_THREADS) * 8;
                v[c] = bf16v8_zero();
                av[c] = bf16v8_zero();
                w[c] = bf16v8_zero();
                if (i < feat) {
                    v[c] = ld_glob8(bg + base + i);
                    av[c] = ld_glob8(ag + base + i);
                    if (gamma) w[c] = ld_glob8(gg + i);
                }
            }
            /* CONSUME. */
            float ss = 0.0f;
#pragma unroll
            for (int c = 0; c < RN_VEC; c++)
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    const float f = bf2f(v[c][j]);
                    ss += f * f;
                }
            const float inv = rsqrtf(block_sum(ss, part) / (float)feat + eps);
#pragma unroll
            for (int c = 0; c < RN_VEC; c++) {
                const unsigned i = (threadIdx.x + (unsigned)c * PLOW_THREADS) * 8;
                if (i < feat) {
                    bf16v8 o;
#pragma unroll
                    for (int j = 0; j < 8; j++) {
                        const float g = gamma ? bf2f(w[c][j]) : 1.0f;
                        o[j] = f2bf((bf2f(av[c][j]) + bf2f(v[c][j]) * inv * g) * scale);
                    }
                    st_glob8(og + base + i, o);
                }
            }
        } else {
            float ss = 0.0f;
            for (unsigned i = threadIdx.x; i < feat; i += PLOW_THREADS) {
                const float x = bf2f(b[base + i]);
                ss += x * x;
            }
            const float inv = rsqrtf(block_sum(ss, part) / (float)feat + eps);
            for (unsigned i = threadIdx.x; i < feat; i += PLOW_THREADS) {
                const float g = gamma ? bf2f(gamma[i]) : 1.0f;
                st_act1(&out[base + i], f2bf((bf2f(a[base + i]) + bf2f(b[base + i]) * inv * g) * scale));
            }
        }
    }
}

/* Qwen/Llama PRE-NORM tail, in ONE packet: the residual add AND the norm that always follows it.
 *
 *     resid = a + b                    (the running residual stream)
 *     out   = RMSNorm(resid, gamma)    (the normed activation the next sublayer reads)
 *
 * where `a` is the incoming residual and `b` the sublayer output (o_proj / down_proj). Every
 * pre-norm transformer has exactly this shape: `residual += sublayer; normed = norm(residual)`.
 * We were emitting it as a RESIDUAL packet then an RMSNORM packet -- two global gates, and in
 * decode each is a single-row op ONE workgroup runs while 255 wait on its counter. Fusing halves
 * both: a and b are read ONCE, the sum r=a+b is held in registers across the reduction, written
 * back as the residual, and the norm never re-reads it.
 *
 * Distinct from d_norm_residual (Gemma's SANDWICH a + RMSNorm(b)): there the norm is on `b` alone
 * and the residual is not re-normed. Here the norm is on the SUM, and BOTH outputs are written.
 *
 * `resid` aliases `a` in the caller (in-place residual), so neither is __restrict__: every read of
 * a is hoisted into registers before any store to resid, so the alias is harmless. */
__device__ void d_add_norm(bf16* __restrict__ out, bf16* resid, const bf16* a,
                           const bf16* __restrict__ b, const bf16* __restrict__ gamma,
                           unsigned rows, unsigned feat, float eps, unsigned slice, unsigned nblk,
                           float* part) {
    const bool fits = (feat <= RN_REG * PLOW_THREADS) && ((feat & 7u) == 0);
    const auto* ag = as_glob(a);
    const auto* bg = as_glob(b);
    const auto* gg = as_glob(gamma);
    auto* og = as_glob(out);
    auto* rg = as_glob(resid);
    for (unsigned row = slice; row < rows; row += nblk) {
        const size_t base = (size_t)row * feat;
        if (fits) {
            /* PRODUCE: residual a, sublayer output b, and the weight -- all issued together. */
            bf16v8 av[RN_VEC], bv[RN_VEC], w[RN_VEC];
#pragma unroll
            for (int c = 0; c < RN_VEC; c++) {
                const unsigned i = (threadIdx.x + (unsigned)c * PLOW_THREADS) * 8;
                av[c] = bf16v8_zero();
                bv[c] = bf16v8_zero();
                w[c] = bf16v8_zero();
                if (i < feat) {
                    av[c] = ld_glob8(ag + base + i);
                    bv[c] = ld_glob8(bg + base + i);
                    if (gamma) w[c] = ld_glob8(gg + i);
                }
            }
            /* CONSUME: r = a + b in registers, reduce on it, then write resid AND its norm. */
            float r[RN_VEC * 8];
            float ss = 0.0f;
#pragma unroll
            for (int c = 0; c < RN_VEC; c++)
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    const float f = bf2f(av[c][j]) + bf2f(bv[c][j]);
                    r[c * 8 + j] = f;
                    ss += f * f;
                }
            const float inv = rsqrtf(block_sum(ss, part) / (float)feat + eps);
#pragma unroll
            for (int c = 0; c < RN_VEC; c++) {
                const unsigned i = (threadIdx.x + (unsigned)c * PLOW_THREADS) * 8;
                if (i < feat) {
                    bf16v8 ro, no;
#pragma unroll
                    for (int j = 0; j < 8; j++) {
                        const float g = gamma ? bf2f(w[c][j]) : 1.0f;
                        const float f = r[c * 8 + j];
                        ro[j] = f2bf(f);
                        no[j] = f2bf(f * inv * g);
                    }
                    st_glob8(rg + base + i, ro);
                    st_glob8(og + base + i, no);
                }
            }
        } else {
            float ss = 0.0f;
            for (unsigned i = threadIdx.x; i < feat; i += PLOW_THREADS) {
                const float f = bf2f(a[base + i]) + bf2f(b[base + i]);
                ss += f * f;
            }
            const float inv = rsqrtf(block_sum(ss, part) / (float)feat + eps);
            for (unsigned i = threadIdx.x; i < feat; i += PLOW_THREADS) {
                const float g = gamma ? bf2f(gamma[i]) : 1.0f;
                const float f = bf2f(a[base + i]) + bf2f(b[base + i]);
                resid[base + i] = f2bf(f);
                st_act1(&out[base + i], f2bf(f * inv * g));
            }
        }
    }
}

/* Gemma SANDWICH tail AND the norm that follows it, in ONE packet (Experiment N1):
 *
 *     resid = (a + RMSNorm(b, gb)) * scale     (the running residual stream — NORM_RESIDUAL)
 *     out   = RMSNorm(resid, gn)               (the normed activation the next sublayer reads)
 *
 * This is the SANDWICH-path successor to d_add_norm. Gemma decode had two adjacent single-workgroup
 * narrow ops -- a NORM_RESIDUAL (post-attn/post-ffn) writing `resid`, then an RMSNORM re-reading
 * `resid` from HBM to make `out` -- each a row reduction ONE workgroup runs while 255 wait on its
 * counter. Fusing deletes a global gate AND a full HBM round trip: `resid` is held in registers
 * across the second reduction instead of being written and immediately re-fetched. It is narrow ->
 * narrow, so it replicates NOTHING (1 consumer workgroup x 1 = 1) -- the fusion the replication law
 * permits. Distinct from d_add_norm (Qwen/Llama PRE-norm: resid = a + b, no post-norm, no scale);
 * here the sublayer output is post-normed by `gb` before the add and the sum is scaled.
 *
 * BIT-EXACT to the NORM_RESIDUAL + RMSNORM pair by construction: the first reduction runs over the
 * same bf16 `b`; the residual is ROUNDED to bf16 (exactly the value NORM_RESIDUAL stored) before the
 * second reduction runs over it, so the round-trip through HBM is reproduced without the traffic.
 *
 * `resid` aliases `a` in the caller (in-place residual), so neither is __restrict__: every read of a
 * is hoisted into registers before any store to resid. `b`/`out` are distinct. */
/* ONE WORKGROUP, 120 TIMES PER TOKEN, AND SPLITTING IT ACROSS CUs DOES NOT PAY. MEASURED.
 *
 * This op's only parallel axis is ROWS and decode has one row, so it runs on a single CU while
 * 255 idle and the ready queue behind it is zero — ~5.9 us x 120 = 0.71 ms/token, 4.1% of the
 * token. It takes the `fits` path at hidden=5376 (RN_REG*PLOW_THREADS = 8192 >= 5376), issuing
 * a, b, gb, gn in one burst and doing BOTH reductions in registers, so there is no scan to
 * eliminate and nothing to fold into the producing GEMV: the cost is one CU moving 42 KB.
 *
 * The only thing that divides 42 KB is more CUs, and the only axis left is FEATURES. That was
 * built and measured: three counter-gated phases (partial sum-of-squares of b -> combine +
 * residual + partial sum-of-squares of resid -> combine + final norm) over rows x k workgroups,
 * with an f32 partial buffer. Gemma-4-31B bf16, ctx 1024, gfx950, interleaved, paired per round,
 * vs the single-packet op:
 *
 *     k=1 (three packets, ONE feature block -- bit-identical arithmetic, NO widening)  +1.28 ms
 *     k=4                                                                             +1.11 ms
 *     k=8                                                                             +1.42 ms
 *
 * Read the k=1 row first: it is the whole result. Two extra packets, with the arithmetic proven
 * bit-identical (token-identical over 48 greedy steps), cost +1.28 ms/token -- ~5.3 us per added
 * packet, nearly TWICE what the op costs in total. Widening then recovers at most ~0.17 ms of
 * that and stops helping past k=4. There is no k at which this wins.
 *
 * Why the packets cost so much when knob-contract 7a priced a gate at <=0.64 us: 7a split ONE
 * wide packet into three that run CONCURRENTLY on disjoint CU sets, so their gate waits and
 * memory ramps overlap. These three are SERIAL — each waits on the last — and decode's DAG is
 * already a 546-deep chain with nothing ready behind a narrow op. A gate is cheap when it
 * separates concurrent work and expensive when it lengthens the critical path. Do not quote the
 * 0.64 us figure for a serial split.
 *
 * A single-packet cross-workgroup reduction would avoid the extra gates, and it is not available:
 * a workgroup would have to spin on a peer's partial, and under the global-queue scheduler the
 * peer is free to be running some other ready packet. That is a deadlock, which is what the
 * counter protocol exists to prevent.
 *
 * The row axis still works and needs nothing: at decode batch B this op is already B workgroups
 * wide (crates/devgen/src/lib.rs, `rows`), so batched decode widens it for free. */
__device__ void d_norm_residual_norm(bf16* __restrict__ out, bf16* resid, const bf16* a,
                                     const bf16* __restrict__ b, const bf16* __restrict__ gb,
                                     const bf16* __restrict__ gn, unsigned rows, unsigned feat,
                                     float eps, float scale, unsigned slice, unsigned nblk,
                                     float* part) {
    const bool fits = (feat <= RN_REG * PLOW_THREADS) && ((feat & 7u) == 0);
    const auto* ag = as_glob(a);
    const auto* bg = as_glob(b);
    const auto* gbg = as_glob(gb);
    const auto* gng = as_glob(gn);
    auto* og = as_glob(out);
    auto* rg = as_glob(resid);
    for (unsigned row = slice; row < rows; row += nblk) {
        const size_t base = (size_t)row * feat;
        if (fits) {
            /* PRODUCE: residual a, sublayer output b, and BOTH weights -- all issued together. */
            bf16v8 av[RN_VEC], bv[RN_VEC], wb[RN_VEC], wn[RN_VEC];
#pragma unroll
            for (int c = 0; c < RN_VEC; c++) {
                const unsigned i = (threadIdx.x + (unsigned)c * PLOW_THREADS) * 8;
                av[c] = bf16v8_zero();
                bv[c] = bf16v8_zero();
                wb[c] = bf16v8_zero();
                wn[c] = bf16v8_zero();
                if (i < feat) {
                    av[c] = ld_glob8(ag + base + i);
                    bv[c] = ld_glob8(bg + base + i);
                    if (gb) wb[c] = ld_glob8(gbg + i);
                    if (gn) wn[c] = ld_glob8(gng + i);
                }
            }
            /* FIRST norm: RMS over the sublayer output b. */
            float ssb = 0.0f;
#pragma unroll
            for (int c = 0; c < RN_VEC; c++)
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    const float f = bf2f(bv[c][j]);
                    ssb += f * f;
                }
            const float invb = rsqrtf(block_sum(ssb, part) / (float)feat + eps);
            /* resid = (a + norm(b)*gb) * scale, ROUNDED to bf16; SECOND reduction runs over the
             * rounded value, exactly reproducing NORM_RESIDUAL's bf16 store + RMSNORM's reload. */
            bf16v8 rv[RN_VEC];
            float ssr = 0.0f;
#pragma unroll
            for (int c = 0; c < RN_VEC; c++) {
                bf16v8 r;
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    const float g = gb ? bf2f(wb[c][j]) : 1.0f;
                    const float f = (bf2f(av[c][j]) + bf2f(bv[c][j]) * invb * g) * scale;
                    r[j] = f2bf(f);
                    const float rf = bf2f(r[j]);
                    ssr += rf * rf;
                }
                rv[c] = r;
            }
            const float invr = rsqrtf(block_sum(ssr, part) / (float)feat + eps);
#pragma unroll
            for (int c = 0; c < RN_VEC; c++) {
                const unsigned i = (threadIdx.x + (unsigned)c * PLOW_THREADS) * 8;
                if (i < feat) {
                    bf16v8 no;
#pragma unroll
                    for (int j = 0; j < 8; j++) {
                        const float g = gn ? bf2f(wn[c][j]) : 1.0f;
                        no[j] = f2bf(bf2f(rv[c][j]) * invr * g);
                    }
                    st_glob8(rg + base + i, rv[c]);
                    st_glob8(og + base + i, no);
                }
            }
        } else {
            float ssb = 0.0f;
            for (unsigned i = threadIdx.x; i < feat; i += PLOW_THREADS) {
                const float f = bf2f(b[base + i]);
                ssb += f * f;
            }
            const float invb = rsqrtf(block_sum(ssb, part) / (float)feat + eps);
            float ssr = 0.0f;
            for (unsigned i = threadIdx.x; i < feat; i += PLOW_THREADS) {
                const float g = gb ? bf2f(gb[i]) : 1.0f;
                const bf16 rb = f2bf((bf2f(a[base + i]) + bf2f(b[base + i]) * invb * g) * scale);
                resid[base + i] = rb;
                const float rf = bf2f(rb);
                ssr += rf * rf;
            }
            const float invr = rsqrtf(block_sum(ssr, part) / (float)feat + eps);
            for (unsigned i = threadIdx.x; i < feat; i += PLOW_THREADS) {
                const float g = gn ? bf2f(gn[i]) : 1.0f;
                st_act1(&out[base + i], f2bf(bf2f(resid[base + i]) * invr * g));
            }
        }
    }
}

#endif /* PLOW_OP_NORM_H */
