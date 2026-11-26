/* op_mamba.cuh — Mamba-2 (Nemotron-H) mixer ops for the sm_120 interpreter, opcodes 240..245.
 *
 * The in_proj / out_proj projections are ordinary FP8 GEMV/GEMM; these ops are the mixer core,
 * decomposed so every one parallelizes over the whole grid:
 *   MAMBA2_CONV        causal depthwise conv + bias + SiLU over xBC (decode: per-slot history;
 *                      prefill: one sequence continuing from its slot's history)
 *   MAMBA2_STEP        decode SSM update + y = C.h + D x (fp16 state, stochastic rounding)
 *   MAMBA2_GATED_NORM  grouped RMSNorm(y * silu(z)) * gamma (gate before the norm)
 *   MAMBA2_CHUNK_STATE prefill phase 1: dt, per-chunk decay cumsum, per-chunk local states
 *   MAMBA2_CHUNK_PASS  prefill phase 2: inter-chunk recurrence from the slot state; entering
 *                      state per chunk (in place) and the final state
 *   MAMBA2_CHUNK_OUT   prefill phase 3: intra-chunk (C B^T o decay) x + C . entering state + D x
 * Semantics follow vLLM MambaMixer2 (causal_conv1d, selective_state_update, mamba_chunk_scan,
 * MixerRMSNormGated): bf16 activations and conv history, f32 math. The slot SSM state is fp16
 * (vLLM --mamba-ssm-cache-dtype float16 with the FlashInfer SSU): decode stores it with stochastic
 * rounding, prefill's final state with round-to-nearest-even (vLLM's chunk-scan store).
 * Prefill pad rows (t >= real rows) are the identity on both states (conv output 0, dt 0) and
 * write zero outputs.
 */
#pragma once
#include "sm120_common.cuh"

static __device__ __forceinline__ float mamba_silu(float x) { return x / (1.0f + __expf(-x)); }
/* vLLM's softplus: log1p(exp(x)) below the threshold 20, identity above. */
static __device__ __forceinline__ float mamba_softplus(float x) {
    return x <= 20.0f ? log1pf(__expf(x)) : x;
}
static __device__ __forceinline__ float mamba_warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
static __device__ __forceinline__ float mamba_bf(const __nv_bfloat16* p, size_t i) {
    return __bfloat162float(p[i]);
}
static __device__ __forceinline__ float mamba_dt(float raw, float bias, float lo, float hi) {
    return fminf(fmaxf(mamba_softplus(raw + bias), lo), hi);
}

/* Slot SSM state element: fp16 (devgen nemotron_h SSM_STATE_BYTES; ops carry it as an
 * immediate and trap on any other width). */
#define MAMBA_STATE_BYTES 2u
static __device__ __forceinline__ float4 mamba_ld_h4(const __half* p) {
    const uint2 v = *(const uint2*)p;
    const float2 a = __half22float2(*(const __half2*)&v.x), b = __half22float2(*(const __half2*)&v.y);
    return make_float4(a.x, a.y, b.x, b.y);
}
static __device__ __forceinline__ void mamba_st_h4_rn(__half* p, float4 s) {
    const __half2 a = __floats2half2_rn(s.x, s.y), b = __floats2half2_rn(s.z, s.w);
    *(uint2*)p = make_uint2(*(const unsigned*)&a, *(const unsigned*)&b);
}

/* The peer's stochastic rounding (FlashInfer mamba/conversion.cuh, the path sm_12x takes):
 * Philox-4x32 = Triton tl.randint4x (key = seed, counter = (offset lo, offset hi, 0, 0)), the
 * per-layer key derived from the step seed by layer_philox_seed (SplitMix64 finalizer), and the
 * software f32 -> f16 SR cvt_rs_f16_sw (13 random bits added to the magnitude, truncate; below
 * the f16 normal range flushes to 0). */
template <int ROUNDS>
static __device__ __forceinline__ uint4 mamba_philox4(unsigned long long key, unsigned long long ctr) {
    unsigned k0 = (unsigned)key, k1 = (unsigned)(key >> 32);
    unsigned c0 = (unsigned)ctr, c1 = (unsigned)(ctr >> 32), c2 = 0u, c3 = 0u;
#pragma unroll
    for (int i = 0; i < ROUNDS; i++) {
        const unsigned a = c0, b = c2;
        c0 = __umulhi(0xCD9E8D57u, b) ^ c1 ^ k0;
        c2 = __umulhi(0xD2511F53u, a) ^ c3 ^ k1;
        c1 = 0xCD9E8D57u * b;
        c3 = 0xD2511F53u * a;
        k0 += 0x9E3779B9u;
        k1 += 0xBB67AE85u;
    }
    return make_uint4(c0, c1, c2, c3);
}
static __device__ __forceinline__ unsigned long long mamba_layer_seed(unsigned seed, unsigned layer) {
    unsigned long long z = (unsigned long long)seed ^ (0x9E3779B97F4A7C15ull * ((unsigned long long)layer + 1ull));
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static __device__ __forceinline__ unsigned mamba_sr_f16(float x, unsigned r) {
    const unsigned bits = __float_as_uint(x);
    const unsigned a = (bits & 0x7FFFFFFFu) + (r & 0x1FFFu);
    const unsigned e = (a >> 23) & 0xFFu, m = a & 0x7FFFFFu;
    const unsigned h = e == 0xFFu ? (m ? 0x7E00u : 0x7C00u)
                     : e > 142u   ? 0x7C00u
                     : e < 113u   ? 0u
                                  : ((e - 112u) << 10) | (m >> 13);
    return ((bits >> 16) & 0x8000u) | h;
}

/* MAMBA2_CONV. `source` (j1): in_stride | in_col << 16 (elements) of xBC inside the in_proj
 * output. history[slot][C][W-1] bf16, oldest first. Decode (prefill == 0): row r is slot r,
 * `rows` = B. Prefill: `rows` = real rows R of a T-row bucket, one sequence (history slot 0 of
 * the launch's table); the thread of (t = 0, c) computes rows 0..W-2 serially (the only rows that
 * read history) and then rewrites the history from rows R-W+1..R-1, so no other thread reads a
 * history it rewrites. */
static __device__ void d_mamba2_conv(__nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ in,
    const __nv_bfloat16* __restrict__ weight, const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ history, const int* __restrict__ active, unsigned channels,
    unsigned width, unsigned rows, unsigned tokens, unsigned prefill, unsigned source,
    unsigned slice, unsigned nblk) {
    if (width < 2 || width > 8) { if (threadIdx.x == 0) __trap(); return; }
    const unsigned is = source ? (source & 0xFFFFu) : channels;
    in += source >> 16;
    const unsigned hw = width - 1;
    if (!prefill) {
        for (unsigned idx = slice * blockDim.x + threadIdx.x; idx < rows * channels;
             idx += nblk * blockDim.x) {
            const unsigned r = idx / channels, c = idx % channels;
            if (active && active[r] <= 0) continue;
            __nv_bfloat16* h = history + (size_t)idx * hw;
            const __nv_bfloat16* w = weight + (size_t)c * width;
            float sum = bias ? mamba_bf(bias, c) : 0.f;
            if (width == 4) {
                /* All loads in one round (the runtime-width loop issued them as dependent
                 * rounds); the same fmaf order. */
                const __nv_bfloat16 h0 = h[0], h1 = h[1], h2 = h[2], x = in[(size_t)r * is + c];
                const float w0 = mamba_bf(w, 0), w1 = mamba_bf(w, 1), w2 = mamba_bf(w, 2), w3 = mamba_bf(w, 3);
                sum = fmaf(__bfloat162float(h0), w0, sum);
                sum = fmaf(__bfloat162float(h1), w1, sum);
                sum = fmaf(__bfloat162float(h2), w2, sum);
                sum = fmaf(__bfloat162float(x), w3, sum);
                h[0] = h1;
                h[1] = h2;
                h[2] = x;
                out[idx] = __float2bfloat16(mamba_silu(sum));
                continue;
            }
            for (unsigned j = 0; j < hw; j++) sum = fmaf(__bfloat162float(h[j]), mamba_bf(w, j), sum);
            const __nv_bfloat16 x = in[(size_t)r * is + c];
            sum = fmaf(__bfloat162float(x), mamba_bf(w, hw), sum);
            for (unsigned j = 0; j + 1 < hw; j++) h[j] = h[j + 1];
            h[hw - 1] = x;
            out[idx] = __float2bfloat16(mamba_silu(sum));
        }
        return;
    }
    if (tokens < hw || rows > tokens) { if (threadIdx.x == 0) __trap(); return; }
    const unsigned head = channels, body = (tokens - hw) * channels;
    for (unsigned idx = slice * blockDim.x + threadIdx.x; idx < head + body; idx += nblk * blockDim.x) {
        if (idx < head) {
            const unsigned c = idx;
            __nv_bfloat16* h = history + (size_t)c * hw;
            const __nv_bfloat16* w = weight + (size_t)c * width;
            const float b = bias ? mamba_bf(bias, c) : 0.f;
            float old[8];
            for (unsigned j = 0; j < hw; j++) old[j] = __bfloat162float(h[j]);
            for (unsigned t = 0; t < hw; t++) {
                float sum = b;
                for (unsigned k = 0; k < width; k++) {
                    const int pos = (int)t - (int)hw + (int)k;
                    const float v = pos < 0 ? old[pos + (int)hw]
                                  : ((unsigned)pos < rows ? mamba_bf(in, (size_t)pos * is + c) : 0.f);
                    sum = fmaf(v, mamba_bf(w, k), sum);
                }
                out[(size_t)t * channels + c] = __float2bfloat16(t < rows ? mamba_silu(sum) : 0.f);
            }
            for (unsigned j = 0; j < hw; j++) {
                const int src = (int)rows - (int)hw + (int)j;
                const float v = src < 0 ? old[src + (int)hw] : mamba_bf(in, (size_t)src * is + c);
                h[j] = __float2bfloat16(v);
            }
        } else {
            const unsigned t = hw + (idx - head) / channels, c = (idx - head) % channels;
            float o = 0.f;
            if (t < rows) {
                const __nv_bfloat16* w = weight + (size_t)c * width;
                float sum = bias ? mamba_bf(bias, c) : 0.f;
                for (unsigned k = 0; k < width; k++)
                    sum = fmaf(mamba_bf(in, (size_t)(t - hw + k) * is + c), mamba_bf(w, k), sum);
                o = mamba_silu(sum);
            }
            out[(size_t)t * channels + c] = __float2bfloat16(o);
        }
    }
}

/* MAMBA2_STEP: one warp per (row b, head h, 8 head-dim rows). Lane owns state columns
 * 4*lane..4*lane+3 (N == 128), so each fp16 state row is one coalesced 256 B warp access and the
 * eight rows' loads are in flight before the math. y written bf16 [B][H*P].
 * dt_source (j1): in_stride | col << 16 of dt inside the in_proj output. The new state is stored
 * with the peer's stochastic rounding: one Philox call per row and lane at the element offset of
 * its first column ((b*H + h)*P + p)*N + 4*lane, element j taking word j; the key derives from
 * the step seed active[seed_word] (runtime-written each step) and the layer. */
static __device__ void d_mamba2_step(__nv_bfloat16* __restrict__ y, const __nv_bfloat16* __restrict__ xbc,
    const __nv_bfloat16* __restrict__ proj, const __nv_bfloat16* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ Dp, const __nv_bfloat16* __restrict__ dt_bias,
    __half* __restrict__ state, const int* __restrict__ active, unsigned heads, unsigned hdim,
    unsigned dstate, unsigned groups, unsigned batch, unsigned dt_source, float dt_lo, float dt_hi,
    unsigned state_bytes, unsigned seed_word, unsigned layer, unsigned slice, unsigned nblk) {
    if (dstate != 128 || hdim % 8 != 0 || groups == 0 || heads % groups != 0 || !dt_source ||
        state_bytes != MAMBA_STATE_BYTES || !active || seed_word < batch) {
        if (threadIdx.x == 0) __trap();
        return;
    }
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    const unsigned d_inner = heads * hdim, conv_dim = d_inner + 2u * groups * dstate;
    const unsigned hpg = heads / groups, ptiles = hdim / 8u;
    const unsigned ds = dt_source & 0xFFFFu, dcol = dt_source >> 16;
    const unsigned long long key = mamba_layer_seed((unsigned)active[seed_word], layer);
    for (unsigned item = slice * warps + warp; item < batch * heads * ptiles; item += nblk * warps) {
        const unsigned b = item / (heads * ptiles), h = (item / ptiles) % heads, p0 = (item % ptiles) * 8u;
        if (active[b] <= 0) continue;
        const unsigned g = h / hpg;
        const size_t off = (((size_t)b * heads + h) * hdim + p0) * dstate + 4u * lane;
        __half* srow = state + off;
        float4 s[8];
#pragma unroll
        for (int i = 0; i < 8; i++) s[i] = mamba_ld_h4(srow + (size_t)i * dstate);
        const __nv_bfloat16* row = xbc + (size_t)b * conv_dim;
        const float dt = mamba_dt(mamba_bf(proj, (size_t)b * ds + dcol + h), mamba_bf(dt_bias, h), dt_lo, dt_hi);
        const float dA = __expf(dt * -__expf(mamba_bf(A_log, h)));
        const float Dh = mamba_bf(Dp, h);
        float Bn[4], Cn[4];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            Bn[j] = mamba_bf(row, d_inner + g * dstate + 4u * lane + j) * dt;
            Cn[j] = mamba_bf(row, d_inner + groups * dstate + g * dstate + 4u * lane + j);
        }
        float x[8], acc[8];
#pragma unroll
        for (int i = 0; i < 8; i++) {
            x[i] = mamba_bf(row, h * hdim + p0 + i);
            s[i].x = fmaf(s[i].x, dA, x[i] * Bn[0]);
            s[i].y = fmaf(s[i].y, dA, x[i] * Bn[1]);
            s[i].z = fmaf(s[i].z, dA, x[i] * Bn[2]);
            s[i].w = fmaf(s[i].w, dA, x[i] * Bn[3]);
            acc[i] = s[i].x * Cn[0] + s[i].y * Cn[1] + s[i].z * Cn[2] + s[i].w * Cn[3];
            const uint4 r = mamba_philox4<5>(key, off + (size_t)i * dstate);
            *(uint2*)(srow + (size_t)i * dstate) =
                make_uint2(mamba_sr_f16(s[i].x, r.x) | mamba_sr_f16(s[i].y, r.y) << 16,
                           mamba_sr_f16(s[i].z, r.z) | mamba_sr_f16(s[i].w, r.w) << 16);
        }
        float v = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const float o = fmaf(Dh, x[i], mamba_warp_sum(acc[i]));
            v = lane == (unsigned)i ? o : v;
        }
        if (lane < 8) y[(size_t)b * d_inner + h * hdim + p0 + lane] = __float2bfloat16(v);
    }
}

/* One (row, group) of MAMBA2_GATED_NORM on one warp, C = gsize / 256 chunks: every chunk's y, z
 * and gamma loads are issued before the math (a runtime trip count left them as dependent
 * rounds); v is recomputed for the output (same expression, same bits) rather than held. */
/* Eight outputs at element `off`: bf16 to out, or with q the e4m3 of bf16 * qinv saturated at
 * +-448 (d_quant_fp8_tensor's arithmetic: the fused static-scale quant of the next GEMM's input). */
static __device__ __forceinline__ void mamba_gn_store(__nv_bfloat16* __restrict__ out, uint8_t* __restrict__ q,
                                                      size_t off, const __nv_bfloat16 (&o)[8], float qinv) {
    if (!q) {
        *(uint4*)(out + off) = *(const uint4*)o;
        return;
    }
    uint2 q8;
    unsigned short* q2 = (unsigned short*)&q8;
#pragma unroll
    for (int j = 0; j < 4; j++)
        q2[j] = pack_fp8_e4m3(fmaxf(-448.0f, fminf(__bfloat162float(o[2 * j]) * qinv, 448.0f)),
                              fmaxf(-448.0f, fminf(__bfloat162float(o[2 * j + 1]) * qinv, 448.0f)));
    *(uint2*)(q + off) = q8;
}

template <unsigned C>
static __device__ __forceinline__ void mamba_gated_norm_row(__nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ yv, const __nv_bfloat16* __restrict__ z,
    const __nv_bfloat16* __restrict__ gamma, unsigned r, unsigned g, unsigned gsize,
    unsigned width, unsigned zs, float eps, unsigned lane, uint8_t* __restrict__ q, float qinv) {
    uint4 ya[C], zb[C], gm[C];
#pragma unroll
    for (unsigned k = 0; k < C; k++) {
        const unsigned col = g * gsize + k * 256u + lane * 8u;
        ya[k] = *(const uint4*)(yv + (size_t)r * width + col);
        zb[k] = *(const uint4*)(z + (size_t)r * zs + col);
        gm[k] = *(const uint4*)(gamma + col);
    }
    auto vf = [&](unsigned k, int j) {
        const __nv_bfloat16* pa = (const __nv_bfloat16*)&ya[k];
        const __nv_bfloat16* pb = (const __nv_bfloat16*)&zb[k];
        return __bfloat162float(pa[j]) * mamba_silu(__bfloat162float(pb[j]));
    };
    float sum = 0.f;
#pragma unroll
    for (unsigned k = 0; k < C; k++) {
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const float f = vf(k, j);
            sum = fmaf(f, f, sum);
        }
    }
    const float inv = rsqrtf(mamba_warp_sum(sum) / (float)gsize + eps);
#pragma unroll
    for (unsigned k = 0; k < C; k++) {
        const unsigned col = g * gsize + k * 256u + lane * 8u;
        const __nv_bfloat16* pg = (const __nv_bfloat16*)&gm[k];
        __nv_bfloat16 o[8];
#pragma unroll
        for (int j = 0; j < 8; j++)
            o[j] = __float2bfloat16(__bfloat162float(__float2bfloat16(vf(k, j) * inv)) * __bfloat162float(pg[j]));
        mamba_gn_store(out, q, (size_t)r * width + col, o, qinv);
    }
}

/* MAMBA2_GATED_NORM: one warp per (row, group). v = y * silu(z) (f32), out =
 * bf16(bf16(v * rsqrt(mean(v^2) + eps)) * gamma), vLLM MixerRMSNormGated (norm_before_gate=False).
 * gsize % 256 == 0 (lane: gsize/256 chunks of 8 bf16). z_source (j1): z_stride | z_col << 16.
 * q (prefill, i3 = 1): out is not written; q = e4m3 of out / qscale[0] (out_proj's static input
 * quant, fused). */
static __device__ void d_mamba2_gated_norm(__nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ yv,
    const __nv_bfloat16* __restrict__ z, const __nv_bfloat16* __restrict__ gamma,
    const int* __restrict__ active, unsigned groups, unsigned gsize, unsigned rows, float eps,
    unsigned z_source, unsigned slice, unsigned nblk, uint8_t* __restrict__ q = nullptr,
    const float* __restrict__ qscale = nullptr) {
    if (gsize % 256u != 0 || gsize > 2048u || (q && !qscale)) { if (threadIdx.x == 0) __trap(); return; }
    const float qinv = q ? 1.0f / qscale[0] : 0.f;
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    const unsigned width = groups * gsize, chunks = gsize / 256u;
    const unsigned zs = z_source ? (z_source & 0xFFFFu) : width;
    z += z_source >> 16;
    for (unsigned item = slice * warps + warp; item < rows * groups; item += nblk * warps) {
        const unsigned r = item / groups, g = item % groups;
        if (active && active[r] <= 0) continue;
        if (chunks == 4u) {
            mamba_gated_norm_row<4>(out, yv, z, gamma, r, g, gsize, width, zs, eps, lane, q, qinv);
            continue;
        }
        float v[8][8];
        float sum = 0.f;
        for (unsigned k = 0; k < chunks; k++) {
            const unsigned col = g * gsize + k * 256u + lane * 8u;
            const uint4 a = *(const uint4*)(yv + (size_t)r * width + col);
            const uint4 b = *(const uint4*)(z + (size_t)r * zs + col);
            const __nv_bfloat16* pa = (const __nv_bfloat16*)&a;
            const __nv_bfloat16* pb = (const __nv_bfloat16*)&b;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const float f = __bfloat162float(pa[j]) * mamba_silu(__bfloat162float(pb[j]));
                v[k][j] = f;
                sum = fmaf(f, f, sum);
            }
        }
        const float inv = rsqrtf(mamba_warp_sum(sum) / (float)gsize + eps);
        for (unsigned k = 0; k < chunks; k++) {
            const unsigned col = g * gsize + k * 256u + lane * 8u;
            const uint4 gm = *(const uint4*)(gamma + col);
            const __nv_bfloat16* pg = (const __nv_bfloat16*)&gm;
            __nv_bfloat16 o[8];
#pragma unroll
            for (int j = 0; j < 8; j++)
                o[j] = __float2bfloat16(__bfloat162float(__float2bfloat16(v[k][j] * inv)) * __bfloat162float(pg[j]));
            mamba_gn_store(out, q, (size_t)r * width + col, o, qinv);
        }
    }
}

/* ---- chunked prefill scan (L = 128 tokens per chunk, P = 64, N = 128, 256 threads) ----
 * Scratch: dtv / cum f32 [H][T] (dt and the inclusive in-chunk cumsum of dt*A), cstate f32
 * [T/L][H][P][N] (local chunk states, overwritten in place by the entering states). */
#define MAMBA_L 128u
#define MAMBA_P 64u
#define MAMBA_N 128u

static __device__ __forceinline__ bool mamba_chunk_shape_ok(unsigned hdim, unsigned dstate,
                                                            unsigned rows, unsigned tokens) {
    return hdim == MAMBA_P && dstate == MAMBA_N && tokens % MAMBA_L == 0 && rows <= tokens &&
           blockDim.x == 256u;
}

/* MAMBA2_CHUNK_STATE: block per (chunk, head). */
static __device__ void d_mamba2_chunk_state(float* __restrict__ cstate, const __nv_bfloat16* __restrict__ xbc,
    const __nv_bfloat16* __restrict__ proj, const __nv_bfloat16* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ dt_bias, float* __restrict__ dtv, float* __restrict__ cum,
    unsigned heads, unsigned hdim, unsigned rows, unsigned tokens, unsigned dstate, unsigned groups,
    float dt_lo, float dt_hi, unsigned dt_source, float* __restrict__ smem, unsigned slice, unsigned nblk) {
    if (!mamba_chunk_shape_ok(hdim, dstate, rows, tokens) || heads % groups != 0 || !dt_source) {
        if (threadIdx.x == 0) __trap();
        return;
    }
    const unsigned tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const unsigned d_inner = heads * MAMBA_P, conv_dim = d_inner + 2u * groups * MAMBA_N;
    const unsigned nchunk = (rows + MAMBA_L - 1) / MAMBA_L, hpg = heads / groups;
    const unsigned ds = dt_source & 0xFFFFu, dcol = dt_source >> 16;
    float* s_w = smem;                 /* [L] decay * dt */
    float* s_part = smem + MAMBA_L;    /* [8] warp scan totals */
    float* xs = smem + MAMBA_L + 32u;  /* [32][P] */
    float* bs = xs + 32u * MAMBA_P;    /* [32][N] */
    for (unsigned item = slice; item < nchunk * heads; item += nblk) {
        const unsigned c = item / heads, h = item % heads, g = h / hpg, t0 = c * MAMBA_L;
        float a = 0.f, dt = 0.f;
        if (tid < MAMBA_L) {
            const unsigned tok = t0 + tid;
            if (tok < rows) {
                dt = mamba_dt(mamba_bf(proj, (size_t)tok * ds + dcol + h), mamba_bf(dt_bias, h), dt_lo, dt_hi);
                a = dt * -__expf(mamba_bf(A_log, h));
            }
            float v = a;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const float u = __shfl_up_sync(0xffffffffu, v, o);
                if (lane >= (unsigned)o) v += u;
            }
            if (lane == 31) s_part[warp] = v;
            a = v;
        }
        __syncthreads();
        if (tid < MAMBA_L) {
            for (unsigned w = 0; w < warp; w++) a += s_part[w];
            dtv[(size_t)h * tokens + t0 + tid] = dt;
            cum[(size_t)h * tokens + t0 + tid] = a;
            s_w[tid] = a;
        }
        __syncthreads();
        const float last = s_w[MAMBA_L - 1];
        __syncthreads();
        if (tid < MAMBA_L) s_w[tid] = __expf(last - s_w[tid]) * dt;
        __syncthreads();
        /* S[p][n] = sum_t w[t] x[t][p] B[t][n]: thread owns p = 4*(tid/16).., n = 8*(tid%16).. */
        const unsigned tp = (tid >> 4) * 4u, tn = (tid & 15u) * 8u;
        float acc[4][8];
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < 8; j++) acc[i][j] = 0.f;
        for (unsigned s0 = 0; s0 < MAMBA_L; s0 += 32u) {
            for (unsigned e = tid; e < 32u * MAMBA_P; e += 256u) {
                const unsigned r = e / MAMBA_P, p = e % MAMBA_P, tok = t0 + s0 + r;
                xs[e] = tok < rows ? s_w[s0 + r] * mamba_bf(xbc, (size_t)tok * conv_dim + h * MAMBA_P + p) : 0.f;
            }
            for (unsigned e = tid; e < 32u * MAMBA_N; e += 256u) {
                const unsigned r = e / MAMBA_N, n = e % MAMBA_N, tok = t0 + s0 + r;
                bs[e] = tok < rows ? mamba_bf(xbc, (size_t)tok * conv_dim + d_inner + g * MAMBA_N + n) : 0.f;
            }
            __syncthreads();
#pragma unroll 4
            for (unsigned r = 0; r < 32u; r++) {
                const float4 xv = *(const float4*)(xs + r * MAMBA_P + tp);
                const float4 b0 = *(const float4*)(bs + r * MAMBA_N + tn);
                const float4 b1 = *(const float4*)(bs + r * MAMBA_N + tn + 4u);
                const float xa[4] = {xv.x, xv.y, xv.z, xv.w};
                const float ba[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
                for (int i = 0; i < 4; i++)
#pragma unroll
                    for (int j = 0; j < 8; j++) acc[i][j] = fmaf(xa[i], ba[j], acc[i][j]);
            }
            __syncthreads();
        }
        float* dst = cstate + (((size_t)c * heads + h) * MAMBA_P + tp) * MAMBA_N + tn;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            *(float4*)(dst + (size_t)i * MAMBA_N) = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
            *(float4*)(dst + (size_t)i * MAMBA_N + 4u) = make_float4(acc[i][4], acc[i][5], acc[i][6], acc[i][7]);
        }
    }
}

/* MAMBA2_CHUNK_PASS: thread per (head, p, 4 state columns). carry = slot state; per chunk the
 * entering state replaces the local state in cstate, carry = exp(cum_last) carry + local.
 * The final carry is the slot's state after the last real row, stored fp16 round-to-nearest. */
static __device__ void d_mamba2_chunk_pass(float* __restrict__ cstate, const float* __restrict__ cum,
    __half* __restrict__ state, unsigned heads, unsigned hdim, unsigned rows, unsigned tokens,
    unsigned dstate, unsigned state_bytes, unsigned slice, unsigned nblk) {
    if (!mamba_chunk_shape_ok(hdim, dstate, rows, tokens) || state_bytes != MAMBA_STATE_BYTES) {
        if (threadIdx.x == 0) __trap();
        return;
    }
    const unsigned nchunk = (rows + MAMBA_L - 1) / MAMBA_L, q = MAMBA_N / 4u;
    const size_t per_chunk = (size_t)heads * MAMBA_P * MAMBA_N;
    for (unsigned idx = slice * blockDim.x + threadIdx.x; idx < heads * MAMBA_P * q; idx += nblk * blockDim.x) {
        const unsigned h = idx / (MAMBA_P * q);
        const size_t off = (size_t)idx * 4u;
        float4 carry = mamba_ld_h4(state + off);
        for (unsigned c = 0; c < nchunk; c++) {
            const float d = __expf(cum[(size_t)h * tokens + c * MAMBA_L + MAMBA_L - 1]);
            float* sp = cstate + c * per_chunk + off;
            const float4 loc = *(const float4*)sp;
            *(float4*)sp = carry;
            carry.x = fmaf(d, carry.x, loc.x);
            carry.y = fmaf(d, carry.y, loc.y);
            carry.z = fmaf(d, carry.z, loc.z);
            carry.w = fmaf(d, carry.w, loc.w);
        }
        mamba_st_h4_rn(state + off, carry);
    }
}

/* MAMBA2_CHUNK_OUT: block per (chunk, head).
 * y[t][p] = exp(cum[t]) sum_n C[t][n] E[p][n]
 *         + sum_{s<=t} (sum_n C[t][n] B[s][n]) exp(cum[t]-cum[s]) dt[s] x[s][p] + D x[t][p].
 * Thread owns t = 8*(tid/16).., p = 4*(tid%16).. of the 128 x 64 output tile. Shared tiles are
 * staged 32 columns of n (or rows of s) at a time, transposed so every inner read is a float4. */
static __device__ void d_mamba2_chunk_out(__nv_bfloat16* __restrict__ y, const __nv_bfloat16* __restrict__ xbc,
    const float* __restrict__ enter, const float* __restrict__ cum, const float* __restrict__ dtv,
    const __nv_bfloat16* __restrict__ Dp, unsigned heads, unsigned hdim, unsigned rows,
    unsigned tokens, unsigned dstate, unsigned groups, float* __restrict__ smem, unsigned slice,
    unsigned nblk) {
    if (!mamba_chunk_shape_ok(hdim, dstate, rows, tokens) || heads % groups != 0) {
        if (threadIdx.x == 0) __trap();
        return;
    }
    const unsigned tid = threadIdx.x;
    const unsigned d_inner = heads * MAMBA_P, conv_dim = d_inner + 2u * groups * MAMBA_N;
    const unsigned nchunk = tokens / MAMBA_L, live = (rows + MAMBA_L - 1) / MAMBA_L, hpg = heads / groups;
    float* s_cum = smem;                         /* [L] */
    float* s_dt = s_cum + MAMBA_L;               /* [L] */
    float* ct = s_dt + MAMBA_L;                  /* [32 n][L t]   C^T tile */
    float* et = ct + 32u * MAMBA_L;              /* [32 n][P]     E^T tile, then x tile [32 s][P] */
    float* bt = et + 32u * MAMBA_P;              /* [32 n][32 s]  B^T tile */
    float* gt = bt + 32u * 32u;                  /* [32 s][L t]   masked G^T tile */
    const unsigned tt = (tid >> 4) * 8u, tp = (tid & 15u) * 4u;
    for (unsigned item = slice; item < nchunk * heads; item += nblk) {
        const unsigned c = item / heads, h = item % heads, g = h / hpg, t0 = c * MAMBA_L;
        if (c >= live) {
            /* Pad-only chunk: zero output rows. */
            for (unsigned e = tid; e < MAMBA_L * MAMBA_P; e += 256u)
                y[(size_t)(t0 + e / MAMBA_P) * d_inner + h * MAMBA_P + e % MAMBA_P] = __float2bfloat16(0.f);
            continue;
        }
        if (tid < MAMBA_L) {
            s_cum[tid] = cum[(size_t)h * tokens + t0 + tid];
            s_dt[tid] = dtv[(size_t)h * tokens + t0 + tid];
        }
        const size_t crow = d_inner + groups * MAMBA_N + g * MAMBA_N, brow = d_inner + g * MAMBA_N;
        float acc[8][4];
#pragma unroll
        for (int i = 0; i < 8; i++)
#pragma unroll
            for (int j = 0; j < 4; j++) acc[i][j] = 0.f;
        /* Off-diagonal: C . E^T over n tiles. */
        const float* E = enter + ((size_t)c * heads + h) * MAMBA_P * MAMBA_N;
        for (unsigned n0 = 0; n0 < MAMBA_N; n0 += 32u) {
            __syncthreads();
            for (unsigned e = tid; e < 32u * MAMBA_L; e += 256u) {
                const unsigned t = e / 32u, n = e % 32u, tok = t0 + t;
                ct[n * MAMBA_L + t] = tok < rows ? mamba_bf(xbc, (size_t)tok * conv_dim + crow + n0 + n) : 0.f;
            }
            for (unsigned e = tid; e < 32u * MAMBA_P; e += 256u) {
                const unsigned p = e / 32u, n = e % 32u;
                et[n * MAMBA_P + p] = E[(size_t)p * MAMBA_N + n0 + n];
            }
            __syncthreads();
#pragma unroll 4
            for (unsigned n = 0; n < 32u; n++) {
                const float4 c0 = *(const float4*)(ct + n * MAMBA_L + tt);
                const float4 c1 = *(const float4*)(ct + n * MAMBA_L + tt + 4u);
                const float4 ev = *(const float4*)(et + n * MAMBA_P + tp);
                const float ca[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
                const float ea[4] = {ev.x, ev.y, ev.z, ev.w};
#pragma unroll
                for (int i = 0; i < 8; i++)
#pragma unroll
                    for (int j = 0; j < 4; j++) acc[i][j] = fmaf(ca[i], ea[j], acc[i][j]);
            }
        }
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const float d = __expf(s_cum[tt + i]);
#pragma unroll
            for (int j = 0; j < 4; j++) acc[i][j] *= d;
        }
        /* Intra-chunk, per source tile of 32 rows s. */
        const unsigned gs = (tid & 7u) * 4u, gtr = (tid >> 3) * 4u; /* G tile: 4 s x 4 t per thread */
        for (unsigned s0 = 0; s0 < MAMBA_L; s0 += 32u) {
            float gacc[4][4];
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) gacc[i][j] = 0.f;
            for (unsigned n0 = 0; n0 < MAMBA_N; n0 += 32u) {
                __syncthreads();
                for (unsigned e = tid; e < 32u * MAMBA_L; e += 256u) {
                    const unsigned t = e / 32u, n = e % 32u, tok = t0 + t;
                    ct[n * MAMBA_L + t] = tok < rows ? mamba_bf(xbc, (size_t)tok * conv_dim + crow + n0 + n) : 0.f;
                }
                for (unsigned e = tid; e < 32u * 32u; e += 256u) {
                    const unsigned s = e / 32u, n = e % 32u, tok = t0 + s0 + s;
                    bt[n * 32u + s] = tok < rows ? mamba_bf(xbc, (size_t)tok * conv_dim + brow + n0 + n) : 0.f;
                }
                __syncthreads();
#pragma unroll 4
                for (unsigned n = 0; n < 32u; n++) {
                    const float4 cv = *(const float4*)(ct + n * MAMBA_L + gtr);
                    const float4 bv = *(const float4*)(bt + n * 32u + gs);
                    const float ca[4] = {cv.x, cv.y, cv.z, cv.w};
                    const float ba[4] = {bv.x, bv.y, bv.z, bv.w};
#pragma unroll
                    for (int i = 0; i < 4; i++)
#pragma unroll
                        for (int j = 0; j < 4; j++) gacc[i][j] = fmaf(ba[i], ca[j], gacc[i][j]);
                }
            }
            /* Mask + decay, store G^T[s][t]; stage x[s][p] into `et`. */
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const unsigned s = s0 + gs + i;
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const unsigned t = gtr + j;
                    gt[(gs + i) * MAMBA_L + t] =
                        s <= t ? gacc[i][j] * __expf(s_cum[t] - s_cum[s]) * s_dt[s] : 0.f;
                }
            }
            for (unsigned e = tid; e < 32u * MAMBA_P; e += 256u) {
                const unsigned s = e / MAMBA_P, p = e % MAMBA_P, tok = t0 + s0 + s;
                et[e] = tok < rows ? mamba_bf(xbc, (size_t)tok * conv_dim + h * MAMBA_P + p) : 0.f;
            }
            __syncthreads();
#pragma unroll 4
            for (unsigned s = 0; s < 32u; s++) {
                const float4 g0 = *(const float4*)(gt + s * MAMBA_L + tt);
                const float4 g1 = *(const float4*)(gt + s * MAMBA_L + tt + 4u);
                const float4 xv = *(const float4*)(et + s * MAMBA_P + tp);
                const float ga[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
                const float xa[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
                for (int i = 0; i < 8; i++)
#pragma unroll
                    for (int j = 0; j < 4; j++) acc[i][j] = fmaf(ga[i], xa[j], acc[i][j]);
            }
        }
        const float Dh = mamba_bf(Dp, h);
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const unsigned tok = t0 + tt + i;
            const size_t base = (size_t)tok * d_inner + h * MAMBA_P + tp;
            __nv_bfloat16 o[4];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const float x = tok < rows ? mamba_bf(xbc, (size_t)tok * conv_dim + h * MAMBA_P + tp + j) : 0.f;
                o[j] = __float2bfloat16(tok < rows ? fmaf(Dh, x, acc[i][j]) : 0.f);
            }
            *(uint2*)(y + base) = *(const uint2*)o;
        }
        __syncthreads();
    }
}
