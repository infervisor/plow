/* op_qwen_gdn.h — the AMD arms for Qwen3.5's Gated DeltaNet block.
 *
 * Ported body-for-body from runtime/nvidia/op_qwen_gdn.cuh, which is the AUTHORITATIVE semantics:
 * the packet layouts, the rounding points and the order of operations below are that file's, not a
 * re-derivation. Where the two differ it is because a 64-lane wave forced it, and every such place
 * is commented.
 *
 * Per head, per token, carrying a state S in [V,K] f32:
 *
 *     S  <-  diag_none( exp(-exp(A_log) * softplus(a + dt_bias)) ) * S            (forget gate)
 *     S  <-  S + beta * (v - S k) (x) k                                          (delta rule)
 *     o  =   S q
 *
 * SAME FAMILY AS KDA, ONE STRUCTURAL DIFFERENCE. KDA's gate is per (head, key-channel) — a vector
 * `g` of length D that op_kda.h carries as a per-chunk log2-domain prefix. Gated DeltaNet's gate is
 * a SCALAR per (token, head). Everything else matches: L2-normalized q/k, the same beta-scaled
 * rank-1 delta rule, the same V-first f32 HBM state, the same [head, V-row] ownership. In FLA's
 * own kernels the two are literally one body under a `USE_GK` flag. So the decode arm here is
 * `d_kda_state_step_t` with a broadcast gate, and a future chunked prefill arm should be
 * `d_kda_chunk_carry_bt64` fed a broadcast gate prefix rather than a second recurrence framework.
 *
 * WHAT IS NOT HERE. `PLOW_DOP_QWEN_GDN_PREFILL` (146) has no arm on EITHER backend: sm_90a
 * dispatches it host-side into a generated CuTe kernel (runtime/nvidia/gdn_prefill.cpp). An AMD
 * prefill needs that adapter plus a chunked body, not a `case` label, and until it exists
 * `refuse_unimplemented_target` in crates/devgen/src/qwen35.rs refuses the emit by name.
 *
 * WAVE 64, NOT 32. The NVIDIA bodies pin `kdim == 128` and `dim == 256` because a 32-lane warp
 * covers them in exactly 4 and 8 registers. Here the lane depth is a TEMPLATE parameter
 * (`PL = dim / PLOW_WAVE`) with a rung dispatch, so the bodies stay generic over geometry and a
 * geometry with no rung refuses loudly instead of running short. The one place the 64-lane wave
 * changes the algorithm rather than the loop bounds is the RoPE pair exchange in
 * `d_qwen_headnorm_rope`; see the note there.
 *
 * All bodies take (slice, nblk) where a standalone kernel would take (blockIdx.x, gridDim.x).
 *
 * TRANSCENDENTALS ARE THE PRECISE ONES. `expf`/`log1pf`, not `__expf` — these bodies are memory
 * bound (the state round-trip dominates), the gate feeds a multiplicative recurrence over a
 * PERSISTENT state where a relative error compounds across steps, and matching the NVIDIA arm is
 * worth more here than the handful of VALU a fast-math exp would save.
 */
#ifndef PLOW_OP_QWEN_GDN_H
#define PLOW_OP_QWEN_GDN_H

#include "amd_common.h"

__device__ __forceinline__ float qwen_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

/* vLLM rounds beta to the projection dtype before the recurrence; the NVIDIA arm reproduces that
 * and so must this one, or the two backends drift on the third bf16 mantissa bit. */
__device__ __forceinline__ float qwen_beta(bf16 raw) { return bf2f(f2bf(qwen_sigmoid(bf2f(raw)))); }

__device__ __forceinline__ float qwen_softplus(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

/* ---- op 136: causal depthwise conv + SiLU over the decode step, with a shifting history ----
 * One thread owns one (row, channel): the history shift below is serial and per-row, so no two
 * threads may touch the same `history` window. */
__device__ void d_qwen_gdn_conv(bf16* out, const bf16* in, const bf16* weight, bf16* history,
                                const int* active, unsigned channels, unsigned width,
                                unsigned batch, unsigned slice, unsigned nblk) {
    if (width < 2) {
        if (threadIdx.x == 0) __builtin_trap();
        return;
    }
    for (unsigned row = slice * blockDim.x + threadIdx.x; row < batch * channels;
         row += nblk * blockDim.x) {
        if (active && active[row / channels] <= 0) continue;
        const unsigned c = row % channels;
        bf16* h = history + (size_t)row * (width - 1);
        const bf16* w = weight + (size_t)c * width;
        float sum = 0.0f;
        for (unsigned j = 0; j + 1 < width; j++) sum = fmaf(bf2f(h[j]), bf2f(w[j]), sum);
        const bf16 x = in[row];
        sum = fmaf(bf2f(x), bf2f(w[width - 1]), sum);
        for (unsigned j = 0; j + 2 < width; j++) h[j] = h[j + 1];
        h[width - 2] = x;
        out[row] = f2bf(sum * qwen_sigmoid(sum));
    }
}

/* PLOW_QWEN_GDN_VROWS — how many V rows of one head a wave owns.
 *
 * Everything above `projection` below is a property of the (slot, head) PAIR, not of the V row:
 * the q/k loads, the two `wave_sum` reductions over them, `sqrtf`+divide twice, the nested
 * `expf(-expf(al) * softplus(dt))` and the `beta` sigmoid. At vdim = 128 the shipped one-row-
 * per-wave mapping recomputes all of it 128 times per head. VROWS > 1 gives a wave a strip of
 * consecutive V rows so that work happens once per strip — the same restructuring
 * `PLOW_NV_GDN_STEP_VROWS8` already carries on the NVIDIA arm, ported rather than reinvented.
 *
 * ARITHMETIC-PRESERVING BY CONSTRUCTION, and that is why it is checkable: every row's arithmetic
 * is unchanged and independent (the recurrence runs along kdim within a row, never across rows),
 * `q[j] *= qs` / `k[j] *= ks` produce the same products wherever they are hoisted to, and each
 * state column still has exactly one writer. Only the row-to-wave mapping moves. `kx gdn_vrows`
 * DEMANDS bit-identity of the bf16 output rather than assuming it, and gets it at VROWS 2/4/8;
 * the f32 state is scored against the f64 oracle instead, where all four arms sit inside the
 * shipped arm's own run-to-run band.
 *
 * IT IS NOT FREE: it divides the launch's wave count by VROWS. At the Qwen3-Next decode shape
 * (hv=48, vdim=128, batch=1) 6144 rows over 304 workgroups x 8 waves is 2.5 rows per wave
 * already, so VROWS=4 leaves 37% of the waves with nothing to do and VROWS=8 leaves 68%. That
 * tradeoff is the measurement, not a detail — see `runtime/bench/amd/kx/exp_gdn_vrows.hip` and
 * the numbers in `docs/amd/instruction-cost-arms-20260908.md`. Default 1 = the shipped mapping. */
#ifndef PLOW_QWEN_GDN_VROWS
#define PLOW_QWEN_GDN_VROWS 1
#endif
#if PLOW_QWEN_GDN_VROWS < 1
#error "PLOW_QWEN_GDN_VROWS must be >= 1"
#endif

/* ---- op 137: the decode recurrence ----
 * `PL` is the per-lane key depth, `kdim / PLOW_WAVE`. A WAVE owns one V row of the V-first state
 * (or, at PLOW_QWEN_GDN_VROWS > 1, a strip of them within ONE head), so no two waves write the
 * same state column and no barrier is needed.
 *
 * State layout is [slot][hv][vdim][kdim] f32 — V-FIRST, exactly as the NVIDIA arm and as
 * op_kda.h's carry. With vdim == kdim a transposed state has the right byte count and the right
 * norm; the stride arithmetic here IS the assertion, so do not "simplify" it. */
template <unsigned PL>
__device__ void d_qwen_gdn_step_t(bf16* out, const bf16* qkv, const bf16* a, const bf16* b,
                                  const void* a_log, const bf16* dt_bias, float* state,
                                  const int* active, unsigned hk, unsigned hv, unsigned kdim,
                                  unsigned vdim, unsigned batch, float scale, float eps,
                                  unsigned alog_f32, unsigned slice, unsigned nblk) {
    const unsigned lane = threadIdx.x & (PLOW_WAVE - 1u), wave = threadIdx.x >> 6;
    const unsigned waves = blockDim.x >> 6;
    const unsigned packed = 2 * hk * kdim + hv * vdim;
#if PLOW_QWEN_GDN_VROWS > 1
    const unsigned tiles = (vdim + PLOW_QWEN_GDN_VROWS - 1u) / PLOW_QWEN_GDN_VROWS;
    for (unsigned tile = slice * waves + wave; tile < batch * hv * tiles; tile += nblk * waves) {
        const unsigned slot = tile / (hv * tiles);
        if (active && active[slot] <= 0) continue;
        const unsigned head = tile / tiles % hv, qhead = head / (hv / hk);
        const unsigned first_v = (tile % tiles) * PLOW_QWEN_GDN_VROWS;
#else
    for (unsigned row = slice * waves + wave; row < batch * hv * vdim; row += nblk * waves) {
        const unsigned slot = row / (hv * vdim);
        if (active && active[slot] <= 0) continue;
        const unsigned head = row / vdim % hv, vcol = row % vdim, qhead = head / (hv / hk);
#endif
        const bf16* x = qkv + (size_t)slot * packed;
        float q[PL], k[PL], h[PL], qq = 0.0f, kk = 0.0f;
#pragma unroll
        for (unsigned j = 0; j < PL; j++) {
            const unsigned d = lane + PLOW_WAVE * j;
            q[j] = bf2f(x[qhead * kdim + d]);
            k[j] = bf2f(x[hk * kdim + qhead * kdim + d]);
            qq += q[j] * q[j];
            kk += k[j] * k[j];
        }
        const float qs = scale / sqrtf(wave_sum(qq) + eps);
        const float ks = 1.0f / sqrtf(wave_sum(kk) + eps);
        const unsigned gate = slot * hv + head;
        const float dt = bf2f(a[gate]) + bf2f(dt_bias[head]);
        const float al = alog_f32 ? ((const float*)a_log)[head] : bf2f(((const bf16*)a_log)[head]);
        const float decay = expf(-expf(al) * qwen_softplus(dt));
        const float beta = qwen_beta(b[gate]);
#if PLOW_QWEN_GDN_VROWS > 1
#pragma unroll
        for (unsigned j = 0; j < PL; j++) {
            q[j] *= qs;
            k[j] *= ks;
        }
/* `unroll 1`: the strip loop carries `state` traffic and two `wave_sum`s per row, so unrolling it
 * buys nothing and multiplies the code size of an already register-tight body. */
#pragma unroll 1
        for (unsigned vcol = first_v; vcol < vdim && vcol < first_v + PLOW_QWEN_GDN_VROWS; vcol++) {
            const unsigned row = ((slot * hv + head) * vdim) + vcol;
#endif
            float projection = 0.0f;
#pragma unroll
            for (unsigned j = 0; j < PL; j++) {
#if PLOW_QWEN_GDN_VROWS == 1
                q[j] *= qs;
                k[j] *= ks;
#endif
                h[j] = state[(size_t)row * kdim + lane + PLOW_WAVE * j] * decay;
                projection += h[j] * k[j];
            }
            const float value = bf2f(x[2 * hk * kdim + head * vdim + vcol]);
            const float delta = (value - wave_sum(projection)) * beta;
            float result = 0.0f;
#pragma unroll
            for (unsigned j = 0; j < PL; j++) {
                h[j] = fmaf(delta, k[j], h[j]);
                state[(size_t)row * kdim + lane + PLOW_WAVE * j] = h[j];
                result += h[j] * q[j];
            }
            result = wave_sum(result);
            if (lane == 0) out[row] = f2bf(result);
#if PLOW_QWEN_GDN_VROWS > 1
        }
#endif
    }
}

/* The rungs. A runtime-bounded `float q[kdim / 64]` would land in scratch (a spill), which is why
 * `PL` is compile-time; a kdim with no rung refuses rather than running one of the others short. */
#define PLOW_QWEN_STEP_RUNGS(CALL_)                                                                \
    switch (kdim) {                                                                                \
        case 64: CALL_(1); break;                                                                  \
        case 128: CALL_(2); break;                                                                 \
        case 192: CALL_(3); break;                                                                 \
        case 256: CALL_(4); break;                                                                 \
        default:                                                                                   \
            if (threadIdx.x == 0) __builtin_trap();                                                \
            return;                                                                                \
    }

__device__ void d_qwen_gdn_step(bf16* out, const bf16* qkv, const bf16* a, const bf16* b,
                                const void* a_log, const bf16* dt_bias, float* state,
                                const int* active, unsigned hk, unsigned hv, unsigned kdim,
                                unsigned vdim, unsigned batch, float scale, float eps,
                                unsigned alog_f32, unsigned slice, unsigned nblk) {
    if (!hk || !hv || hv % hk || !vdim || (kdim % PLOW_WAVE)) {
        if (threadIdx.x == 0) __builtin_trap();
        return;
    }
#define PLOW_QWEN_STEP_CALL(PL_)                                                                   \
    d_qwen_gdn_step_t<PL_>(out, qkv, a, b, a_log, dt_bias, state, active, hk, hv, kdim, vdim,      \
                           batch, scale, eps, alog_f32, slice, nblk)
    PLOW_QWEN_STEP_RUNGS(PLOW_QWEN_STEP_CALL)
#undef PLOW_QWEN_STEP_CALL
}

/* ---- op 138: RMSNorm over the head dim, gated by SiLU(z) ---- */
__device__ void d_qwen_gated_norm(bf16* out, const bf16* core, const bf16* z, const bf16* gamma,
                                  const int* active, unsigned heads, unsigned dim, unsigned batch,
                                  float eps, unsigned slice, unsigned nblk) {
    const unsigned lane = threadIdx.x & (PLOW_WAVE - 1u), wave = threadIdx.x >> 6;
    const unsigned waves = blockDim.x >> 6;
    for (unsigned row = slice * waves + wave; row < batch * heads; row += nblk * waves) {
        if (active && active[row / heads] <= 0) continue;
        const size_t off = (size_t)row * dim;
        float sum = 0.0f;
        for (unsigned d = lane; d < dim; d += PLOW_WAVE) {
            const float v = bf2f(core[off + d]);
            sum += v * v;
        }
        const float inv = rsqrtf(wave_sum(sum) / dim + eps);
        for (unsigned d = lane; d < dim; d += PLOW_WAVE) {
            const float v = bf2f(core[off + d]) * inv * bf2f(gamma[d]);
            const float gate = bf2f(z[off + d]);
            out[off + d] = f2bf(v * gate * qwen_sigmoid(gate));
        }
    }
}

/* ---- op 139: split a [heads][2 * dim] projection into q and its gate ---- */
__device__ void d_qwen_q_gate_split(bf16* q, bf16* gate, const bf16* packed, const int* active,
                                    unsigned heads, unsigned dim, unsigned batch, unsigned slice,
                                    unsigned nblk) {
    const unsigned cols = heads * dim;
    for (unsigned i = slice * blockDim.x + threadIdx.x; i < batch * cols;
         i += nblk * blockDim.x) {
        if (active && active[i / cols] <= 0) continue;
        const size_t src = (size_t)(i / dim) * 2 * dim + i % dim;
        q[i] = packed[src];
        gate[i] = packed[src + dim];
    }
}

/* ---- op 140: elementwise sigmoid gate, with vLLM's bf16 round on the gate ---- */
__device__ void d_qwen_sigmoid_gate(bf16* out, const bf16* in, const bf16* gate, const int* active,
                                    unsigned cols, unsigned batch, unsigned slice, unsigned nblk) {
    for (unsigned i = slice * blockDim.x + threadIdx.x; i < batch * cols;
         i += nblk * blockDim.x) {
        if (active && active[i / cols] <= 0) continue;
        out[i] = f2bf(bf2f(in[i]) * qwen_beta(gate[i]));
    }
}

/* ---- op 141: RMSNorm with a (gamma + offset) weight ---- */
__device__ void d_qwen_rmsnorm(bf16* out, const bf16* in, const bf16* gamma, const int* active,
                               unsigned dim, unsigned batch, float eps, float offset,
                               unsigned slice, unsigned nblk) {
    const unsigned lane = threadIdx.x & (PLOW_WAVE - 1u), wave = threadIdx.x >> 6;
    const unsigned waves = blockDim.x >> 6;
    for (unsigned row = slice * waves + wave; row < batch; row += nblk * waves) {
        if (active && active[row] <= 0) continue;
        const size_t base = (size_t)row * dim;
        float sum = 0.0f;
        for (unsigned d = lane; d < dim; d += PLOW_WAVE) {
            const float x = bf2f(in[base + d]);
            sum += x * x;
        }
        const float inv = rsqrtf(wave_sum(sum) / dim + eps);
        for (unsigned d = lane; d < dim; d += PLOW_WAVE)
            out[base + d] = f2bf(bf2f(in[base + d]) * inv * (bf2f(gamma[d]) + offset));
    }
}

/* ---- op 142: per-head RMSNorm then partial RoPE, optionally scattered into a KV ring ----
 *
 * THE ONE BODY A 64-LANE WAVE CHANGES. The NVIDIA arm holds element `d = lane + 32 * j` and rotates
 * the pair (x[0], x[1]) — which at warp 32 is (d, d + 32), the two halves of a 64-wide rotary
 * section, in the SAME lane. At wave 64 register 0 already spans [0, 64), so the partner of lane
 * `l` is lane `l ^ 32` and the pair must be exchanged across lanes. `__shfl_xor(_, 32, 64)` is that
 * exchange; lanes below 32 hold the low half, lanes at or above it hold the high half, and both
 * read the same cos/sin entry `lane & 31`. This is the transliteration amd_common.h:1075 warns is
 * not available. */
template <unsigned PL>
__device__ void d_qwen_headnorm_rope_t(bf16* out, const bf16* in, const bf16* gamma,
                                       const float* cos, const float* sin, const int* positions,
                                       const int* active, unsigned heads, unsigned dim,
                                       unsigned rotary, unsigned batch, unsigned context,
                                       unsigned normalize, float eps, float offset, unsigned slice,
                                       unsigned nblk, unsigned prefill) {
    const unsigned lane = threadIdx.x & (PLOW_WAVE - 1u), wave = threadIdx.x >> 6;
    const unsigned waves = blockDim.x >> 6;
    for (unsigned row = slice * waves + wave; row < batch * heads; row += nblk * waves) {
        const unsigned slot = row / heads, head = row % heads;
        if (active && active[slot] <= 0) continue;
        const int pos = positions ? positions[slot] : 0;
        if (pos < 0 || (context && (unsigned)pos >= context)) {
            if (lane == 0) __builtin_trap();
            continue;
        }
        float x[PL], sum = 0.0f;
#pragma unroll
        for (unsigned j = 0; j < PL; j++) {
            x[j] = bf2f(in[(size_t)row * dim + lane + j * PLOW_WAVE]);
            sum += x[j] * x[j];
        }
        if (normalize) {
            const float inv = rsqrtf(wave_sum(sum) / dim + eps);
#pragma unroll
            for (unsigned j = 0; j < PL; j++) {
                const float g = gamma ? bf2f(gamma[lane + j * PLOW_WAVE]) + offset : 1.0f;
                x[j] = bf2f(f2bf(x[j] * inv * g));
            }
        }
        if (rotary) {
            const unsigned half = rotary >> 1;
            /* Unconditional: a shuffle under divergence leaves the inactive lanes' contribution
             * undefined, so every lane exchanges and only the rotary section applies the result. */
            const float partner = __shfl_xor(x[0], half, PLOW_WAVE);
            if (lane < rotary) {
                /* vLLM stores its Qwen rotary cache in the model BF16 dtype; the round is load
                 * bearing — the tables are f32 in HBM and the reference reads them as bf16. */
                const float c = bf2f(f2bf(cos[(size_t)pos * half + (lane & (half - 1u))]));
                const float s = bf2f(f2bf(sin[(size_t)pos * half + (lane & (half - 1u))]));
                x[0] = lane < half ? x[0] * c - partner * s : x[0] * c + partner * s;
            }
        }
        const size_t dest = context
            ? ((size_t)(prefill ? 0 : slot) * heads + head) * context * dim + (size_t)pos * dim
            : (size_t)row * dim;
#pragma unroll
        for (unsigned j = 0; j < PL; j++)
            out[dest + lane + j * PLOW_WAVE] = f2bf(x[j]);
    }
}

__device__ void d_qwen_headnorm_rope(bf16* out, const bf16* in, const bf16* gamma,
                                     const float* cos, const float* sin, const int* positions,
                                     const int* active, unsigned heads, unsigned dim,
                                     unsigned rotary, unsigned batch, unsigned context,
                                     unsigned normalize, float eps, float offset, unsigned slice,
                                     unsigned nblk, unsigned prefill) {
    /* `rotary <= PLOW_WAVE` keeps the rotated section inside register 0, which is what makes the
     * partner a single `lane ^ (rotary/2)` shuffle. Qwen3.5 uses rotary = 64 on a 256-wide head. */
    if ((dim % PLOW_WAVE) || (rotary && (rotary > PLOW_WAVE || (rotary & (rotary - 1u))))) {
        if (threadIdx.x == 0) __builtin_trap();
        return;
    }
#define PLOW_QWEN_HNR_CALL(PL_)                                                                    \
    d_qwen_headnorm_rope_t<PL_>(out, in, gamma, cos, sin, positions, active, heads, dim, rotary,   \
                                batch, context, normalize, eps, offset, slice, nblk, prefill)
    switch (dim) {
        case 64: PLOW_QWEN_HNR_CALL(1); break;
        case 128: PLOW_QWEN_HNR_CALL(2); break;
        case 192: PLOW_QWEN_HNR_CALL(3); break;
        case 256: PLOW_QWEN_HNR_CALL(4); break;
        default:
            if (threadIdx.x == 0) __builtin_trap();
            return;
    }
#undef PLOW_QWEN_HNR_CALL
}

/* ---- op 143: the same conv over a whole prefill chunk ----
 * One thread owns a channel for the WHOLE chunk and updates the history only after consuming it,
 * so the history window is never read after a partial write. */
__device__ void d_qwen_gdn_conv_prefill(bf16* out, const bf16* in, const bf16* weight,
                                        bf16* history, unsigned channels, unsigned width,
                                        unsigned tokens, unsigned slice, unsigned nblk) {
    if (width < 2 || !tokens) {
        if (threadIdx.x == 0) __builtin_trap();
        return;
    }
    for (unsigned c = slice * blockDim.x + threadIdx.x; c < channels; c += nblk * blockDim.x) {
        const bf16* w = weight + (size_t)c * width;
        bf16* h = history + (size_t)c * (width - 1);
        for (unsigned t = 0; t < tokens; t++) {
            float sum = 0.0f;
            for (unsigned j = 0; j < width; j++) {
                const int source = (int)t + (int)j - (int)(width - 1);
                const float x =
                    source < 0 ? bf2f(h[t + j]) : bf2f(in[(size_t)source * channels + c]);
                sum = fmaf(x, bf2f(w[j]), sum);
            }
            out[(size_t)t * channels + c] = f2bf(sum * qwen_sigmoid(sum));
        }
        for (unsigned j = 0; j + 1 < width; j++) {
            const int source = (int)tokens + (int)j - (int)(width - 1);
            h[j] = source < 0 ? h[tokens + j] : in[(size_t)source * channels + c];
        }
    }
}

/* ---- op 144: split the packed prefill projection into L2-normalized q/k and a plain v ---- */
template <unsigned PL>
__device__ void d_qwen_gdn_qkv_prep_t(bf16* q, bf16* k, bf16* v, const bf16* packed, unsigned hk,
                                      unsigned hv, unsigned kd, unsigned vd, unsigned tokens,
                                      float eps, unsigned slice, unsigned nblk) {
    const unsigned lane = threadIdx.x & (PLOW_WAVE - 1u), wave = threadIdx.x >> 6;
    const unsigned waves = blockDim.x >> 6;
    const unsigned channels = 2 * hk * kd + hv * vd;
    for (unsigned row = slice * waves + wave; row < tokens * hk; row += nblk * waves) {
        const unsigned t = row / hk, head = row % hk;
        float qv[PL], kv[PL], qq = 0.0f, kk = 0.0f;
#pragma unroll
        for (unsigned j = 0; j < PL; j++) {
            const unsigned d = lane + PLOW_WAVE * j;
            qv[j] = bf2f(packed[(size_t)t * channels + head * kd + d]);
            kv[j] = bf2f(packed[(size_t)t * channels + hk * kd + head * kd + d]);
            qq += qv[j] * qv[j];
            kk += kv[j] * kv[j];
        }
        const float qinv = rsqrtf(wave_sum(qq) + eps), kinv = rsqrtf(wave_sum(kk) + eps);
#pragma unroll
        for (unsigned j = 0; j < PL; j++) {
            const size_t dest = (size_t)row * kd + lane + PLOW_WAVE * j;
            q[dest] = f2bf(qv[j] * qinv);
            k[dest] = f2bf(kv[j] * kinv);
        }
    }
    for (unsigned i = slice * blockDim.x + threadIdx.x; i < tokens * hv * vd;
         i += nblk * blockDim.x)
        v[i] = packed[(size_t)(i / (hv * vd)) * channels + 2 * hk * kd + i % (hv * vd)];
}

__device__ void d_qwen_gdn_qkv_prep(bf16* q, bf16* k, bf16* v, const bf16* packed, unsigned hk,
                                    unsigned hv, unsigned kd, unsigned vd, unsigned tokens,
                                    float eps, unsigned slice, unsigned nblk) {
    if (!hk || !hv || !vd || (kd % PLOW_WAVE)) {
        if (threadIdx.x == 0) __builtin_trap();
        return;
    }
#define PLOW_QWEN_PREP_CALL(PL_)                                                                   \
    d_qwen_gdn_qkv_prep_t<PL_>(q, k, v, packed, hk, hv, kd, vd, tokens, eps, slice, nblk)
    switch (kd) {
        case 64: PLOW_QWEN_PREP_CALL(1); break;
        case 128: PLOW_QWEN_PREP_CALL(2); break;
        case 192: PLOW_QWEN_PREP_CALL(3); break;
        case 256: PLOW_QWEN_PREP_CALL(4); break;
        default:
            if (threadIdx.x == 0) __builtin_trap();
            return;
    }
#undef PLOW_QWEN_PREP_CALL
}

/* ---- op 145: the prefill gate pair, alpha (the scalar forget gate) and beta ----
 * `alpha` is the LINEAR decay here, not a log — the chunked prefill consumer takes it that way.
 * op 137 recomputes the same quantity inline for the decode step; the two must agree, which is why
 * both go through `qwen_softplus`/`qwen_beta` rather than open-coding the expression twice. */
__device__ void d_qwen_gdn_gate_prep(float* alpha, float* beta, const bf16* a, const bf16* b,
                                     const bf16* alog, const bf16* bias, unsigned heads,
                                     unsigned tokens, unsigned slice, unsigned nblk) {
    for (unsigned i = slice * blockDim.x + threadIdx.x; i < tokens * heads;
         i += nblk * blockDim.x) {
        const unsigned head = i % heads;
        const float dt = bf2f(a[i]) + bf2f(bias[head]);
        alpha[i] = expf(-expf(bf2f(alog[head])) * qwen_softplus(dt));
        beta[i] = qwen_beta(b[i]);
    }
}

#endif /* PLOW_OP_QWEN_GDN_H */
