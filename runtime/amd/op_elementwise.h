/* op_elementwise.h — bandwidth-bound pointwise ops (CDNA).
 *
 * All of these are memory-bound, so they are written as flat, fully-coalesced strided loops
 * over 16-byte accesses; there is nothing to tile.
 *
 * The 16-byte accesses go through ld_glob8/st_glob8, not __builtin_memcpy. A `bf16*` from the
 * interpreter's tensor table is a GENERIC, align-2 pointer, so a 16-byte memcpy off it is
 * compiled exactly as written -- eight 2-byte `flat_load_ushort`s. See amd_common.h. */
#ifndef PLOW_OP_ELEMENTWISE_H
#define PLOW_OP_ELEMENTWISE_H

#include "amd_common.h"

/* THE FAST SPELLINGS WERE TRIED HERE AND DELIBERATELY NOT KEPT. DO NOT RE-DERIVE THEM.
 *
 * `fast_tanhf` for the gelu and `x * rcp(1 + __expf(-x))` for the silu (both in amd_common.h,
 * both exhaustively swept in runtime/tests/situ_identity_gfx950_test.hip and correct to one bf16
 * ulp) cut these from 38 and 26 VALU to 13 and 5, and they DO make `d_glu` faster — measured at
 * K3's 18432 width, silu 0.0232 -> 0.0213 ms and gelu 0.0240 -> 0.0226 ms, about 7%. They are
 * still not here, for two reasons that only appear once the numbers are put next to each other:
 *
 *   1. THE WIN IS ~0.001 ms PER LAYER. `d_glu` is a 6 B/element streaming pass already running
 *      near its bandwidth roofline; 7% of it is not a number that survives contact with a MoE
 *      layer that costs 10 ms.
 *   2. THESE TWO ARE NOT LOCAL. `act_gate_only` inlines them into FIVE op_gemm.h epilogues
 *      (:668, :1229, :1670, :2156, :3046), which are register-tight MFMA and GEMV kernels of
 *      exactly the kind where this sweep measured an epilogue edit COSTING more than its VALU
 *      saved: the same substitution applied to op_moe.h's `moe_act` made the A4W4 MoE prefill
 *      0.18 ms SLOWER on an arm that never even executes it (see `moe_act`'s note in op_moe.h).
 *      Proving those five neutral is a bench this tree does not have.
 *
 * And `act_silu` is GLM-5.2's dense-FFN SwiGLU, i.e. the shipping model's numerics, for a win
 * that rounds to zero. Left exactly as it was. */
__device__ __forceinline__ float act_gelu_tanh(float x) {
    /* gelu_pytorch_tanh: 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715 x^3))) */
    const float c = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(c));
}
__device__ __forceinline__ float act_silu(float x) { return x / (1.0f + __expf(-x)); }

/* out = (a + b) * scale, or with `pre`: out = (pre + bf16(a + b)) * scale.
 *
 * `scale` exists to absorb Gemma's per-layer `layer_scalar`: HF applies
 * `hidden_states *= layer_scalar` to the whole residual stream at the end of the
 * block, which is exactly `(r + f) * layer_scalar` — so the SECOND residual add
 * folds it in for free. Pass 1.0f for the first add.
 *
 * THE THREE-INPUT FORM (`pre != nullptr`), and why a chained pair of adds is worth an operand.
 * Kimi-K3's MoE tail ends in two Residuals that ONLY each other read:
 *
 *     ffn = up_latent + shared_down          (the two FFN halves)
 *     x   = prefix    + ffn                  (the block output)
 *
 * At decode both are `vec8_cus(7168)` = TWO workgroups (crates/devgen/src/k3.rs), so this is two
 * serial 2-CU packets back to back on a chain with nothing ready behind either, 92 times per token
 * — the same shape, and the same price, as the AttnRes/RMSNorm pair in op_k3.h. `pre` folds them:
 * `a`/`b` are the inner pair and `pre` the outer residual, `ffn` is never written, and one packet
 * and a full 7168-wide HBM round trip disappear.
 *
 * BIT-EXACT, and the `f2bf` in the middle is the whole reason: the inner sum is ROUNDED to bf16
 * before the outer add, which is precisely the value the deleted packet stored and the surviving
 * one re-read. The inner add takes NO scale — the K3 emitter asserts that it was 1.0f — so the
 * packet's one `scale` slot still means what it meant. */
__device__ void d_residual(bf16* __restrict__ out, const bf16* __restrict__ a,
                           const bf16* __restrict__ b, unsigned n, float scale,
                           unsigned slice, unsigned nblk, const bf16* __restrict__ pre = nullptr) {
    const unsigned stride = nblk * PLOW_THREADS * 8;
    const auto* ag = as_glob(a);
    const auto* bg = as_glob(b);
    const auto* pg = as_glob(pre);
    auto* og = as_glob(out);
    for (unsigned i = (slice * PLOW_THREADS + threadIdx.x) * 8; i < n; i += stride) {
        if (i + 8 <= n) {
            const bf16v8 va = ld_glob8(ag + i), vb = ld_glob8(bg + i);
            /* Issued with a/b, not after them: three operands, ONE round trip. */
            const bf16v8 vp = pre ? ld_glob8(pg + i) : bf16v8_zero();
            bf16v8 vo;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const float s = bf2f(va[j]) + bf2f(vb[j]);
                vo[j] = pre ? f2bf((bf2f(vp[j]) + bf2f(f2bf(s))) * scale) : f2bf(s * scale);
            }
            st_glob8(og + i, vo);
        } else {
            for (unsigned j = i; j < n; j++) {
                const float s = bf2f(a[j]) + bf2f(b[j]);
                st_act1(&out[j], pre ? f2bf((bf2f(pre[j]) + bf2f(f2bf(s))) * scale) : f2bf(s * scale));
            }
        }
    }
}

/* Gated MLP: act(gate) * up.
 *
 * Gemma is GeGLU (gelu_pytorch_tanh), NOT SwiGLU. `act` selects so the same op
 * serves Llama/Qwen-style silu gating. */
enum { PLOW_ACT_GELU_TANH_ = 0, PLOW_ACT_SILU_ = 1, PLOW_ACT_SITU_ = 2 };

/* The GATE-ONLY half of a GLU epilogue, for every fused path that computes
 * `act(gate) * up` and therefore CANNOT express Kimi-K3's `situ`.
 *
 * Same argument as `moe_act`'s in op_moe.h, and the same answer. situ is
 * `beta*tanh(g/beta)*sigmoid(g) * lbeta*tanh(u/lbeta)`: it transforms the UP
 * branch as well as the gate, so the expression SHAPE is `A(g)*B(u)`, not
 * `act(g)*u`. A gate-only path handed `act = 2` used to fall through this
 * function's `else` and return gelu_tanh(g)*u — finite, correctly shaped, and
 * the wrong model, with the error growing in the tail of `u`.
 *
 * There is no device trap and this interpreter's dispatch `default:` is a silent
 * NOP, so a NaN is the loudest primitive available: it reaches the residual on
 * the next op. Any epilogue that must actually SUPPORT situ takes the betas and
 * calls a pair-form helper (`moe_glu`, `k3_situ_gate`/`k3_situ_up`) instead of
 * this. Every caller here is one that does not. */
__device__ __forceinline__ float act_gate_only(float g, unsigned act) {
    if (act == PLOW_ACT_SITU_) return __builtin_nanf("");
    return (act == PLOW_ACT_SILU_) ? act_silu(g) : act_gelu_tanh(g);
}

__device__ void d_glu(bf16* __restrict__ out, const bf16* __restrict__ gate,
                      const bf16* __restrict__ up, unsigned n, unsigned act,
                      unsigned slice, unsigned nblk) {
    const unsigned stride = nblk * PLOW_THREADS * 8;
    const auto* gg = as_glob(gate);
    const auto* ug = as_glob(up);
    auto* og = as_glob(out);
    for (unsigned i = (slice * PLOW_THREADS + threadIdx.x) * 8; i < n; i += stride) {
        if (i + 8 <= n) {
            const bf16v8 vg = ld_glob8(gg + i), vu = ld_glob8(ug + i);
            bf16v8 vo;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const float g = bf2f(vg[j]);
                const float a = act_gate_only(g, act);
                vo[j] = f2bf(a * bf2f(vu[j]));
            }
            st_glob8(og + i, vo);
        } else {
            for (unsigned j = i; j < n; j++) {
                const float g = bf2f(gate[j]);
                const float a = act_gate_only(g, act);
                st_act1(&out[j], f2bf(a * bf2f(up[j])));
            }
        }
    }
}

/* Final logit softcapping: cap * tanh(x / cap). Gemma 4 uses cap = 30 on the
 * lm_head output. There is NO attention-logit softcapping in the text tower. */
__device__ void d_softcap(bf16* __restrict__ out, const bf16* __restrict__ x, unsigned n,
                          float cap, unsigned slice, unsigned nblk) {
    const float inv = 1.0f / cap;
    const auto* xg = as_glob(x);
    auto* og = as_glob(out);
    const unsigned stride = nblk * PLOW_THREADS * 8;
    for (unsigned i = (slice * PLOW_THREADS + threadIdx.x) * 8; i < n; i += stride) {
        if (i + 8 <= n) {
            const bf16v8 v = ld_glob8(xg + i);
            bf16v8 o;
#pragma unroll
            for (int j = 0; j < 8; j++) o[j] = f2bf(cap * tanhf(bf2f(v[j]) * inv));
            st_glob8(og + i, o);
        } else {
            for (unsigned j = i; j < n; j++) st_act1(&out[j], f2bf(cap * tanhf(bf2f(x[j]) * inv)));
        }
    }
}

/* Embedding gather + scale.
 *
 * `scale` must be the BF16-ROUNDED sqrt(hidden): 73.5 for the 31B (not 73.3212)
 * and 62.0 for the 12B (not 61.9677). HF downcasts the normalizer to the weight
 * dtype before multiplying and the rounding is observable in the logits. */
__device__ void d_embed(bf16* __restrict__ out, const bf16* __restrict__ table,
                        const int* __restrict__ ids, unsigned ntok, unsigned hidden,
                        float scale, unsigned slice, unsigned nblk) {
    const auto* tg = as_glob(table);
    auto* og = as_glob(out);
    for (unsigned t = slice; t < ntok; t += nblk) {
        const size_t src = (size_t)ids[t] * hidden, dst = (size_t)t * hidden;
        if ((hidden & 7u) == 0) {
            for (unsigned i = threadIdx.x * 8; i < hidden; i += PLOW_THREADS * 8) {
                const bf16v8 v = ld_glob8(tg + src + i);
                bf16v8 o;
#pragma unroll
                for (int j = 0; j < 8; j++) o[j] = f2bf(bf2f(v[j]) * scale);
                st_glob8(og + dst + i, o);
            }
        } else {
            for (unsigned i = threadIdx.x; i < hidden; i += PLOW_THREADS)
                st_act1(&out[dst + i], f2bf(bf2f(table[src + i]) * scale));
        }
    }
}

/* Greedy argmax over the logit row, as an unsigned MAX over a packed key.
 *
 *   [63:32] an order-preserving u32 image of the bf16 value
 *   [31:0]  ~index
 *
 * The bf16 image is the standard trick: flip every bit of a negative, set the sign bit of a
 * positive, and the unsigned ordering of the result matches the float ordering. Storing the
 * COMPLEMENT of the index makes the same `max` break ties toward the LOWEST index, which is
 * what the host's `x > best` loop did. So the whole reduction is one u64 max -- no compare-
 * and-swap, no second value to carry alongside. */
__device__ __forceinline__ unsigned long long amax_pack(bf16 b, unsigned i) {
    const unsigned key = (b & 0x8000u) ? (unsigned)(unsigned short)~b : (unsigned)(b | 0x8000u);
    return ((unsigned long long)key << 32) | (unsigned long long)(~i);
}

__device__ __forceinline__ unsigned long long wave_max_u64(unsigned long long v) {
#pragma unroll
    for (int off = 32; off > 0; off >>= 1) {
        const unsigned long long o = __shfl_xor(v, off, PLOW_WAVE);
        v = o > v ? o : v;
    }
    return v;
}

__device__ __forceinline__ unsigned long long block_max_u64(unsigned long long v,
                                                            unsigned long long* part) {
    const unsigned lane = threadIdx.x & 63, wave = threadIdx.x >> 6;
    v = wave_max_u64(v);
    if (lane == 0) part[wave] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long r = part[0];
#pragma unroll
        for (unsigned w = 1; w < PLOW_WAVES; w++) r = part[w] > r ? part[w] : r;
        part[0] = r;
    }
    __syncthreads();
    return part[0];
}

/* Per-block partial. `part` needs no zeroing: every block writes its own slot unconditionally.
 *
 * BATCH>1 (mirrors runtime/nvidia/op_elementwise.cuh): logits are [n_batch][n] and each sequence
 * gets its OWN argmax — one token per sequence, no cross-sequence bleed. `part` is
 * [n_batch][nblk]; the packed index stays inside the sequence's own [0,n) vocab row.
 * n_batch == 0/1 is byte-identical (part[slice]). Without this the batched decode dispatch
 * sampled sequence 0 only and left every other sequence's `in.ids` slot untouched. */
__device__ void d_argmax(unsigned long long* __restrict__ part, const bf16* __restrict__ x,
                         unsigned n, unsigned n_batch, unsigned slice, unsigned nblk,
                         unsigned long long* lds) {
    const auto* xg = as_glob(x);
    const unsigned B = n_batch ? n_batch : 1u;
    for (unsigned b = 0; b < B; b++) {
        const auto* xb = xg + (size_t)b * n;
        unsigned long long best = 0;
        for (unsigned i = slice * PLOW_THREADS + threadIdx.x; i < n; i += nblk * PLOW_THREADS) {
            const unsigned long long p = amax_pack(xb[i], i);
            best = p > best ? p : best;
        }
        best = block_max_u64(best, lds);
        if (threadIdx.x == 0) st_act<unsigned long long>(&as_glob(part)[(size_t)b * nblk + slice], best);
    }
}

/* Fold the per-block partials and write each sequence's token id where the next step's EMBED
 * reads it. BATCH>1: ids[b] gets sequence b's token; `part` is [n_batch][nparts].
 * n_batch == 0/1 is byte-identical (ids[0] from part[0..nparts)). */
__device__ void d_argmax_fin(int* __restrict__ ids, const unsigned long long* __restrict__ part,
                             unsigned nparts, unsigned n_batch, unsigned slice, unsigned nblk) {
    const auto* pg = as_glob(part);
    const unsigned B = n_batch ? n_batch : 1u;
    const unsigned wave = threadIdx.x >> 6, lane = threadIdx.x & 63u;
    const unsigned nwave = PLOW_THREADS >> 6;
    for (unsigned b = slice * nwave + wave; b < B; b += nblk * nwave) {
        const auto* pb = pg + (size_t)b * nparts;
        unsigned long long best = 0;
        for (unsigned i = lane; i < nparts; i += PLOW_WAVE)
            best = pb[i] > best ? pb[i] : best;
        best = wave_max_u64(best);
        if (lane == 0)
            st_act<int>(&as_glob(ids)[b], (int)~(unsigned)(best & 0xFFFFFFFFull));
    }
}

#endif /* PLOW_OP_ELEMENTWISE_H */
