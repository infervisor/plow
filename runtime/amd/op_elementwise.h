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

__device__ __forceinline__ float act_gelu_tanh(float x) {
    /* gelu_pytorch_tanh: 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715 x^3))) */
    const float c = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(c));
}
__device__ __forceinline__ float act_silu(float x) { return x / (1.0f + __expf(-x)); }

/* out = (a + b) * scale.
 *
 * `scale` exists to absorb Gemma's per-layer `layer_scalar`: HF applies
 * `hidden_states *= layer_scalar` to the whole residual stream at the end of the
 * block, which is exactly `(r + f) * layer_scalar` — so the SECOND residual add
 * folds it in for free. Pass 1.0f for the first add. */
__device__ void d_residual(bf16* __restrict__ out, const bf16* __restrict__ a,
                           const bf16* __restrict__ b, unsigned n, float scale,
                           unsigned slice, unsigned nblk) {
    const unsigned stride = nblk * PLOW_THREADS * 8;
    const auto* ag = as_glob(a);
    const auto* bg = as_glob(b);
    auto* og = as_glob(out);
    for (unsigned i = (slice * PLOW_THREADS + threadIdx.x) * 8; i < n; i += stride) {
        if (i + 8 <= n) {
            const bf16v8 va = ld_glob8(ag + i), vb = ld_glob8(bg + i);
            bf16v8 vo;
#pragma unroll
            for (int j = 0; j < 8; j++) vo[j] = f2bf((bf2f(va[j]) + bf2f(vb[j])) * scale);
            st_glob8(og + i, vo);
        } else {
            for (unsigned j = i; j < n; j++)
                out[j] = f2bf((bf2f(a[j]) + bf2f(b[j])) * scale);
        }
    }
}

/* Gated MLP: act(gate) * up.
 *
 * Gemma is GeGLU (gelu_pytorch_tanh), NOT SwiGLU. `act` selects so the same op
 * serves Llama/Qwen-style silu gating. */
enum { PLOW_ACT_GELU_TANH_ = 0, PLOW_ACT_SILU_ = 1 };

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
                const float a = (act == PLOW_ACT_SILU_) ? act_silu(g) : act_gelu_tanh(g);
                vo[j] = f2bf(a * bf2f(vu[j]));
            }
            st_glob8(og + i, vo);
        } else {
            for (unsigned j = i; j < n; j++) {
                const float g = bf2f(gate[j]);
                const float a = (act == PLOW_ACT_SILU_) ? act_silu(g) : act_gelu_tanh(g);
                out[j] = f2bf(a * bf2f(up[j]));
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
            for (unsigned j = i; j < n; j++) out[j] = f2bf(cap * tanhf(bf2f(x[j]) * inv));
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
                out[dst + i] = f2bf(bf2f(table[src + i]) * scale);
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

/* Per-block partial. `part` needs no zeroing: every block writes its own slot unconditionally. */
__device__ void d_argmax(unsigned long long* __restrict__ part, const bf16* __restrict__ x,
                         unsigned n, unsigned slice, unsigned nblk, unsigned long long* lds) {
    const auto* xg = as_glob(x);
    unsigned long long best = 0;
    for (unsigned i = slice * PLOW_THREADS + threadIdx.x; i < n; i += nblk * PLOW_THREADS) {
        const unsigned long long p = amax_pack(xg[i], i);
        best = p > best ? p : best;
    }
    best = block_max_u64(best, lds);
    if (threadIdx.x == 0) as_glob(part)[slice] = best;
}

/* Fold the per-block partials and write the token id where the next step's EMBED will read it. */
__device__ void d_argmax_fin(int* __restrict__ ids, const unsigned long long* __restrict__ part,
                             unsigned nparts, unsigned slice) {
    if (slice != 0 || threadIdx.x != 0) return;
    const auto* pg = as_glob(part);
    unsigned long long best = 0;
    for (unsigned i = 0; i < nparts; i++) best = pg[i] > best ? pg[i] : best;
    as_glob(ids)[0] = (int)~(unsigned)(best & 0xFFFFFFFFull);
}

#endif /* PLOW_OP_ELEMENTWISE_H */
