#pragma once
#include <cuda_bf16.h>
#include <cmath>

static __device__ __forceinline__ float qwen_warp_sum(float v) {
#pragma unroll
    for (int d = 16; d; d >>= 1) v += __shfl_down_sync(0xffffffffu, v, d);
    return __shfl_sync(0xffffffffu, v, 0);
}
// Eight independent qwen_warp_sum reductions with their shuffle stages interleaved: the same
// tree per value (bit-identical), one pipelined pass instead of eight dependent ones.
static __device__ __forceinline__ void qwen_warp_sum8(float (&v)[8]) {
#pragma unroll
    for (int d = 16; d; d >>= 1) {
#pragma unroll
        for (int r = 0; r < 8; r++) v[r] += __shfl_down_sync(0xffffffffu, v[r], d);
    }
#pragma unroll
    for (int r = 0; r < 8; r++) v[r] = __shfl_sync(0xffffffffu, v[r], 0);
}
// Lanes 0..15 and 16..31 reduce separately; every lane of a half gets its half's total.
static __device__ __forceinline__ float qwen_half_sum(float v) {
#pragma unroll
    for (int d = 8; d; d >>= 1) v += __shfl_xor_sync(0xffffffffu, v, d);
    return v;
}
// bf16 pairs in a 32-bit word (low half = lower address); bf16 -> f32 is the exact shift
// __bfloat162float performs, f32 -> bf16 rounds like __float2bfloat16. Keeps 16-byte vectors in
// registers without taking their address.
static __device__ __forceinline__ float qwen_bf_lo(unsigned w) { return __uint_as_float(w << 16); }
static __device__ __forceinline__ float qwen_bf_hi(unsigned w) { return __uint_as_float(w & 0xffff0000u); }
static __device__ __forceinline__ unsigned qwen_bf_pack(float lo, float hi) {
    return (unsigned)__bfloat16_as_ushort(__float2bfloat16(lo)) |
        ((unsigned)__bfloat16_as_ushort(__float2bfloat16(hi)) << 16);
}
static __device__ __forceinline__ void qwen_bf_unpack8(const uint4 v, float (&f)[8]) {
    f[0] = qwen_bf_lo(v.x); f[1] = qwen_bf_hi(v.x); f[2] = qwen_bf_lo(v.y); f[3] = qwen_bf_hi(v.y);
    f[4] = qwen_bf_lo(v.z); f[5] = qwen_bf_hi(v.z); f[6] = qwen_bf_lo(v.w); f[7] = qwen_bf_hi(v.w);
}
static __device__ __forceinline__ uint4 qwen_bf_pack8(const float (&f)[8]) {
    return make_uint4(qwen_bf_pack(f[0], f[1]), qwen_bf_pack(f[2], f[3]),
        qwen_bf_pack(f[4], f[5]), qwen_bf_pack(f[6], f[7]));
}
// Sum of squares of a 128-wide row held as four contiguous elements per lane (lane L: 4L..4L+3),
// accumulated exactly as the lane-strided loops do: lane L sums elements L + 32j in order (the
// elements come by shuffle), then the caller runs the same warp tree. Bit-identical to the
// 2-byte walks; a half-warp sum of the contiguous elements flipped two of 46 decisive FP32-gate
// positions. The MUFU sigmoid (__expf / __fdividef) flipped them too and is not used.
static __device__ __forceinline__ float qwen_strided_sq4(const float (&f)[4], unsigned lane) {
    float s = 0.f;
#pragma unroll
    for (unsigned j = 0; j < 4; j++) {
        const unsigned src = (lane >> 2) + 8u * j, p = lane & 3u;
        const float x0 = __shfl_sync(0xffffffffu, f[0], src);
        const float x1 = __shfl_sync(0xffffffffu, f[1], src);
        const float x2 = __shfl_sync(0xffffffffu, f[2], src);
        const float x3 = __shfl_sync(0xffffffffu, f[3], src);
        const float x = p == 0u ? x0 : p == 1u ? x1 : p == 2u ? x2 : x3;
        s += x * x;
    }
    return s;
}
// The same sum over a row still packed as a uint2 (lane L: elements 4L..4L+3): the bf16 pairs
// move as 32-bit words (two shuffles per tap instead of four float shuffles and three selects)
// and each lane converts its own element after the move. bf16 -> f32 is exact, so the squares,
// the per-lane order and the warp tree are those of qwen_strided_sq4: bit-identical.
static __device__ __forceinline__ float qwen_strided_sq4_pk(const uint2 v, unsigned lane) {
    float s = 0.f;
    const unsigned p = lane & 3u;
#pragma unroll
    for (unsigned j = 0; j < 4; j++) {
        const unsigned src = (lane >> 2) + 8u * j;
        const unsigned wx = __shfl_sync(0xffffffffu, v.x, src), wy = __shfl_sync(0xffffffffu, v.y, src);
        const unsigned w = (p & 2u) ? wy : wx;
        const float x = (p & 1u) ? qwen_bf_hi(w) : qwen_bf_lo(w);
        s += x * x;
    }
    return s;
}
static __device__ __forceinline__ void qwen_bf_unpack4(const uint2 v, float (&f)[4]) {
    f[0] = qwen_bf_lo(v.x); f[1] = qwen_bf_hi(v.x); f[2] = qwen_bf_lo(v.y); f[3] = qwen_bf_hi(v.y);
}
static __device__ __forceinline__ uint2 qwen_bf_pack4(const float (&f)[4]) {
    return make_uint2(qwen_bf_pack(f[0], f[1]), qwen_bf_pack(f[2], f[3]));
}
// x / (1 + expf(-x)) (SiLU) and 1 / (1 + expf(-x)) (sigmoid) as the IEEE fast path ptxas emits
// for div.rn.f32 (MUFU.RCP, one Newton step, the quotient, one remainder correction) written out
// without the slow-path branch: the compiler wraps every quotient in a convergence barrier and
// a conditional call, and the barriers serialized the conv prefill's eight channel chains
// (light conv 71 us per 1024-row site at 11% DRAM / 17% SM, ~14 us byte floor; H100 27B). The
// slow path only rescales huge denominators; here a denominator above 2^125 (x < -86.6) and
// its numerator are scaled by 2^-64 (exact) into the fast path's range, and a denominator of
// +inf (expf overflow, x < -88.73) gives the division's x * 0. Bit-identical to the division on
// every bf16 input and 2^28 fp32 samples (silu_test, H100).
static __device__ __forceinline__ float qwen_div_pos(float x, float d) {
    const float s = d >= 0x1p125f ? 0x1p-64f : 1.f;
    const float ds = fminf(d, 3.4028235e38f) * s, xs = x * s;
    float r;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(ds));
    r = fmaf(r, fmaf(-ds, r, 1.f), r);
    const float q = xs * r;
    const float out = fmaf(r, fmaf(-ds, q, xs), q);
    return d == INFINITY ? x * 0.f : out;
}
static __device__ __forceinline__ float qwen_silu(float x) {
    const float out = qwen_div_pos(x, 1.f + expf(-x));
    return x == 0.f || x == INFINITY ? x : out;
}
static __device__ __forceinline__ float qwen_sigmoid(float x) {
    return qwen_div_pos(1.f, 1.f + expf(-x));
}

static __device__ void d_qwen_gdn_conv(__nv_bfloat16* out, const __nv_bfloat16* in,
    const __nv_bfloat16* weight, __nv_bfloat16* history, const int* active,
    unsigned channels, unsigned width, unsigned batch, unsigned slice, unsigned nblk) {
    if (width < 2) { if (threadIdx.x == 0) __trap(); return; }
    for (unsigned row = slice * blockDim.x + threadIdx.x; row < batch * channels;
         row += nblk * blockDim.x) {
        if (active && active[row / channels] <= 0) continue;
        const unsigned c = row % channels;
        __nv_bfloat16* h = history + (size_t)row * (width - 1);
        const __nv_bfloat16* w = weight + (size_t)c * width;
        float sum = 0.f;
        for (unsigned j = 0; j + 1 < width; j++)
            sum = fmaf(__bfloat162float(h[j]), __bfloat162float(w[j]), sum);
        const __nv_bfloat16 x = in[row];
        sum = fmaf(__bfloat162float(x), __bfloat162float(w[width - 1]), sum);
        for (unsigned j = 0; j + 2 < width; j++) h[j] = h[j + 1];
        h[width - 2] = x;
        out[row] = __float2bfloat16(sum / (1.f + expf(-sum)));
    }
}

// Default on: H100 27B decode, ms at B1/B16 ctx 1K, one-row body 26.9/39.5 vs this 26.6/34.8.
#ifndef PLOW_NV_GDN_STEP_VROWS8
#define PLOW_NV_GDN_STEP_VROWS8 1
#endif
#if PLOW_NV_GDN_STEP_VROWS8 != 0 && PLOW_NV_GDN_STEP_VROWS8 != 1
#error "PLOW_NV_GDN_STEP_VROWS8 must be 0 or 1"
#endif
// The 8-row body stages its tile's state rows (8 x kdim fp32, 4 KB) in the warp's slice of the
// shared arena by cp.async before the gate math, so the eight loads are in flight together
// instead of one dependent round trip per row. Element ownership and reduction order are those
// of the direct loads, so the outputs are bit-identical.
#ifndef PLOW_NV_GDN_STEP_SMEM
#define PLOW_NV_GDN_STEP_SMEM 1
#endif
// Folded residual+norm rungs keep the normalized rows in the arena for the tensor-core walk
// (interp GEMV / GEMV_GLU arms); needs the mma walk at every folded rung, i.e. its B=1 arm too.
// Off when the K-split walk (PLOW_NV_GEMV_K8) serves those rungs: it is worth 0.9 ms at B1 on
// the 27B against the staging's 0.2, and it reads its activations from global memory.
#ifndef PLOW_NV_QWEN_FOLD_STAGED
#define PLOW_NV_QWEN_FOLD_STAGED (PLOW_NV_GEMV_MMA && PLOW_NV_GEMV_MMA_B1 && !PLOW_NV_GEMV_K8)
#endif

__device__ __forceinline__ void qwen_cp16(void* smem, const void* gmem) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}

// `stage`/`stage_floats`: the shared arena and its capacity; the body loads straight from the
// state when the arena cannot hold warps x 8 x kdim floats (or is null).
// `TWO_PHASE`: the staged tile's eight rows in two batched phases (standalone gdn_ride entry
// only: inside the decode interpreter the 24 live accumulators spilled, stack 320 -> 832 B,
// B1 step 23.9 -> 24.9 ms; H100 27B build65).
template <bool TWO_PHASE = false>
static __device__ void d_qwen_gdn_step(__nv_bfloat16* out, const __nv_bfloat16* qkv,
    const __nv_bfloat16* a, const __nv_bfloat16* b, const void* a_log,
    const __nv_bfloat16* dt_bias, float* state, const int* active,
    unsigned hk, unsigned hv, unsigned kdim, unsigned vdim, unsigned batch,
    float scale, float eps, unsigned alog_f32, unsigned slice, unsigned nblk,
    float* stage = nullptr, unsigned stage_floats = 0,
    const int* slot_tab = nullptr, unsigned slot_stride = 0) {
    if (!hk || !hv || hv % hk || kdim != 128 || !vdim) {
        if (threadIdx.x == 0) __trap(); return;
    }
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const unsigned warps = blockDim.x >> 5;
    const unsigned packed = 2 * hk * kdim + hv * vdim;
    // Row i's state lives in slot i, or in slot slot_tab[i * slot_stride] (a packed launch's
    // riders: rows 0..n, the requests' slots).
    auto state_slot = [&](unsigned slot) {
        return slot_tab ? (unsigned)slot_tab[(size_t)slot * slot_stride] : slot;
    };
#if PLOW_NV_GDN_STEP_VROWS8
    // A warp owns 8 consecutive V rows of one head: the q/k normalization and gate math run once
    // per tile and the rows follow serially. H100 27B decode, ms at B1/B16 ctx 1K: one-row body
    // 26.9/39.5, this 26.6/34.8. Issuing all 8 state-row loads before the first reduction
    // (h[8][4] in registers) pushed the decode entry to the 255-register cap with 304 B of
    // stack and measured 27.4/35.7 — slower at every rung, so the serial form stays.
    const unsigned tiles = (vdim + 7u) / 8u;
#if PLOW_NV_GDN_STEP_SMEM
    // Prefetching the next tile into a second slice measured the same (B32 46.35 vs 46.14 ms),
    // so one slice per warp.
    float* buf = (stage && (size_t)warps * 8u * kdim <= stage_floats) ? stage + (size_t)warp * 8u * kdim : nullptr;
#else
    float* buf = nullptr;
#endif
    for (unsigned tile = slice * warps + warp; tile < batch * hv * tiles; tile += nblk * warps) {
        const unsigned slot = tile / (hv * tiles);
        if (active && active[slot] <= 0) continue;
        const unsigned head = tile / tiles % hv, first_v = (tile % tiles) * 8u;
        const unsigned qhead = head / (hv / hk), sslot = state_slot(slot);
        const __nv_bfloat16* x = qkv + (size_t)slot * packed;
        if (buf) {
            // The previous tile's lanes may still be reading the slice.
            __syncwarp();
            const unsigned nrows = vdim - first_v < 8u ? vdim - first_v : 8u;
            const size_t row0 = ((size_t)sslot * hv + head) * vdim + first_v;
            for (unsigned r = 0; r < nrows; r++)
                qwen_cp16(buf + r * kdim + lane * 4u, state + (row0 + r) * kdim + lane * 4u);
            asm volatile("cp.async.commit_group;\n" ::);
        }
        float q[4], k[4], h[4], qq = 0.f, kk = 0.f;
#pragma unroll
        for (unsigned j = 0; j < 4; j++) {
            unsigned d = lane + 32 * j;
            q[j] = __bfloat162float(x[qhead * kdim + d]);
            k[j] = __bfloat162float(x[hk * kdim + qhead * kdim + d]);
            qq += q[j] * q[j]; kk += k[j] * k[j];
        }
        const float qs = scale / sqrtf(qwen_warp_sum(qq) + eps);
        const float ks = 1.f / sqrtf(qwen_warp_sum(kk) + eps);
        const unsigned gate = slot * hv + head;
        const float dt = __bfloat162float(a[gate]) + __bfloat162float(dt_bias[head]);
        const float softplus = dt > 20.f ? dt : log1pf(expf(dt));
        const float al = alog_f32 ? ((const float*)a_log)[head] : __bfloat162float(((const __nv_bfloat16*)a_log)[head]);
        const float decay = expf(-expf(al) * softplus);
        // vLLM's packed decode rounds beta to the projection dtype before recurrence.
        const float beta = __bfloat162float(__float2bfloat16(1.f / (1.f + expf(-__bfloat162float(b[gate])))));
#pragma unroll
        for (unsigned j = 0; j < 4; j++) { q[j] *= qs; k[j] *= ks; }
        if (buf) {
            asm volatile("cp.async.wait_group 0;\n" ::: "memory");
            __syncwarp();
        }
        if (TWO_PHASE && buf) {
            // The staged tile in two phases over all eight rows (the rows of one time step are
            // independent): the projections, one interleaved reduction, the deltas; then the
            // updates, one interleaved reduction, the outputs. Serial rows ran two dependent
            // reductions each, 76 us per layer at B16 vs a ~30 us byte floor. Same per-element
            // arithmetic and reduction tree as the serial loop: bit-identical (ride 1008+16:
            // per rider 0.615 -> 0.540 ms).
            const unsigned nrows = vdim - first_v < 8u ? vdim - first_v : 8u;
            const unsigned row0 = (slot * hv + head) * vdim + first_v;
            const size_t srow0 = ((size_t)sslot * hv + head) * vdim + first_v;
            float proj[8], delta[8], result[8];
#pragma unroll
            for (unsigned r = 0; r < 8; r++) {
                proj[r] = 0.f;
                if (r < nrows) {
                    const float* src = buf + r * kdim;
#pragma unroll
                    for (unsigned j = 0; j < 4; j++) proj[r] += (src[lane + 32 * j] * decay) * k[j];
                }
            }
            qwen_warp_sum8(proj);
#pragma unroll
            for (unsigned r = 0; r < 8; r++) {
                const float value = r < nrows
                    ? __bfloat162float(x[2 * hk * kdim + head * vdim + first_v + r]) : 0.f;
                delta[r] = (value - proj[r]) * beta;
            }
#pragma unroll
            for (unsigned r = 0; r < 8; r++) {
                result[r] = 0.f;
                if (r < nrows) {
                    const float* src = buf + r * kdim;
                    float* dst = state + (srow0 + r) * kdim;
#pragma unroll
                    for (unsigned j = 0; j < 4; j++) {
                        const float hj = fmaf(delta[r], k[j], src[lane + 32 * j] * decay);
                        dst[lane + 32 * j] = hj;
                        result[r] += hj * q[j];
                    }
                }
            }
            qwen_warp_sum8(result);
            if (lane == 0) {
#pragma unroll
                for (unsigned r = 0; r < 8; r++)
                    if (r < nrows) out[row0 + r] = __float2bfloat16(result[r]);
            }
            continue;
        }
#pragma unroll 1
        for (unsigned vcol = first_v; vcol < vdim && vcol < first_v + 8u; vcol++) {
            const unsigned row = (slot * hv + head) * vdim + vcol;
            const size_t srow = ((size_t)sslot * hv + head) * vdim + vcol;
            const float* src = buf ? buf + (vcol - first_v) * kdim : state + srow * kdim;
            float projection = 0.f;
#pragma unroll
            for (unsigned j = 0; j < 4; j++) {
                h[j] = src[lane + 32 * j] * decay;
                projection += h[j] * k[j];
            }
            const float value = __bfloat162float(x[2 * hk * kdim + head * vdim + vcol]);
            const float delta = (value - qwen_warp_sum(projection)) * beta;
            float result = 0.f;
#pragma unroll
            for (unsigned j = 0; j < 4; j++) {
                h[j] = fmaf(delta, k[j], h[j]);
                state[srow * kdim + lane + 32 * j] = h[j];
                result += h[j] * q[j];
            }
            result = qwen_warp_sum(result);
            if (lane == 0) out[row] = __float2bfloat16(result);
        }
    }
#else
    // A warp owns one V row of the V-first state, so no state writers overlap.
    for (unsigned row = slice * warps + warp; row < batch * hv * vdim; row += nblk * warps) {
        const unsigned slot = row / (hv * vdim);
        if (active && active[slot] <= 0) continue;
        const unsigned head = row / vdim % hv, vcol = row % vdim, qhead = head / (hv / hk);
        const size_t srow = ((size_t)state_slot(slot) * hv + head) * vdim + vcol;
        const __nv_bfloat16* x = qkv + (size_t)slot * packed;
        float q[4], k[4], h[4], qq = 0.f, kk = 0.f;
#pragma unroll
        for (unsigned j = 0; j < 4; j++) {
            unsigned d = lane + 32 * j;
            q[j] = __bfloat162float(x[qhead * kdim + d]);
            k[j] = __bfloat162float(x[hk * kdim + qhead * kdim + d]);
            qq += q[j] * q[j]; kk += k[j] * k[j];
        }
        const float qs = scale / sqrtf(qwen_warp_sum(qq) + eps);
        const float ks = 1.f / sqrtf(qwen_warp_sum(kk) + eps);
        const unsigned gate = slot * hv + head;
        const float dt = __bfloat162float(a[gate]) + __bfloat162float(dt_bias[head]);
        const float softplus = dt > 20.f ? dt : log1pf(expf(dt));
        const float al = alog_f32 ? ((const float*)a_log)[head] : __bfloat162float(((const __nv_bfloat16*)a_log)[head]);
        const float decay = expf(-expf(al) * softplus);
        // vLLM's packed decode rounds beta to the projection dtype before recurrence.
        const float beta = __bfloat162float(__float2bfloat16(1.f / (1.f + expf(-__bfloat162float(b[gate])))));
        float projection = 0.f;
#pragma unroll
        for (unsigned j = 0; j < 4; j++) {
            q[j] *= qs; k[j] *= ks;
            h[j] = state[srow * kdim + lane + 32 * j] * decay;
            projection += h[j] * k[j];
        }
        const float value = __bfloat162float(x[2 * hk * kdim + head * vdim + vcol]);
        const float delta = (value - qwen_warp_sum(projection)) * beta;
        float result = 0.f;
#pragma unroll
        for (unsigned j = 0; j < 4; j++) {
            h[j] = fmaf(delta, k[j], h[j]);
            state[srow * kdim + lane + 32 * j] = h[j];
            result += h[j] * q[j];
        }
        result = qwen_warp_sum(result);
        if (lane == 0) out[row] = __float2bfloat16(result);
    }
#endif
}

// PF: the prefill fast path (the prefill interpreter and the decode object's light_pf, which
// instantiates it on its own register/stack budget); the decode interpreter stays generic.
template <bool PF>
static __device__ void d_qwen_gated_norm(__nv_bfloat16* out, const __nv_bfloat16* core,
    const __nv_bfloat16* z, const __nv_bfloat16* gamma, const int* active,
    unsigned heads, unsigned dim, unsigned batch, float eps, unsigned slice, unsigned nblk) {
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    if constexpr (PF) {
    if (dim == 128) {
        // Prefill objects only (the decode rows are few and its entry is at the register cap):
        // eight-byte loads (lane L holds elements 4L..4L+3), four rows per warp step with the
        // eight loads (core and z) in flight. The 2-byte lane-strided two-row walk ran at 21% of
        // the byte floor (10.2 ms per 48 norms at 4096 rows, floor 2.2). Same per-element math
        // and the generic loop's accumulation order (qwen_strided_sq4): bit-identical.
        const unsigned total = batch * heads;
        constexpr unsigned R = 4;
        float gm[4];
        qwen_bf_unpack4(*(const uint2*)(gamma + lane * 4u), gm);
        for (unsigned row0 = (slice * warps + warp) * R; row0 < total; row0 += nblk * warps * R) {
            uint2 c2[R], z2[R];
            bool on[R];
#pragma unroll
            for (unsigned r = 0; r < R; r++) {
                const unsigned row = row0 + r;
                on[r] = row < total && (!active || active[row / heads] > 0);
                const size_t off = (size_t)row * dim + lane * 4u;
                c2[r] = on[r] ? *(const uint2*)(core + off) : make_uint2(0u, 0u);
                z2[r] = on[r] ? *(const uint2*)(z + off) : make_uint2(0u, 0u);
            }
#pragma unroll
            for (unsigned r = 0; r < R; r++) {
                float v[4], g[4];
                qwen_bf_unpack4(c2[r], v);
                qwen_bf_unpack4(z2[r], g);
                const float inv = rsqrtf(qwen_warp_sum(qwen_strided_sq4_pk(c2[r], lane)) / dim + eps);
#pragma unroll
                for (int j = 0; j < 4; j++) v[j] = v[j] * inv * gm[j] * qwen_silu(g[j]);
                if (on[r]) *(uint2*)(out + (size_t)(row0 + r) * dim + lane * 4u) = qwen_bf_pack4(v);
            }
        }
        return;
    }
    } else {
    if (dim == 128) {
        // Same lane ownership and accumulation order as the generic loop (bit-identical), with
        // the four loads per array in flight together: the strided loop ran one dependent load
        // per step (0.6 ms per op at 4096 rows).
        for (unsigned row = slice * warps + warp; row < batch * heads; row += nblk * warps) {
            if (active && active[row / heads] <= 0) continue;
            const size_t off = (size_t)row * dim;
            float v[4], sum = 0.f;
#pragma unroll
            for (int j = 0; j < 4; j++) { v[j] = __bfloat162float(core[off + lane + 32 * j]); sum += v[j] * v[j]; }
            const float inv = rsqrtf(qwen_warp_sum(sum) / dim + eps);
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const unsigned d = lane + 32 * j;
                const float g = __bfloat162float(z[off + d]);
                out[off + d] = __float2bfloat16(v[j] * inv * __bfloat162float(gamma[d]) * qwen_silu(g));
            }
        }
        return;
    }
    }
    for (unsigned row = slice * warps + warp; row < batch * heads; row += nblk * warps) {
        if (active && active[row / heads] <= 0) continue;
        const size_t off = (size_t)row * dim;
        float sum = 0.f;
        for (unsigned d = lane; d < dim; d += 32) {
            float v = __bfloat162float(core[off + d]); sum += v * v;
        }
        const float inv = rsqrtf(qwen_warp_sum(sum) / dim + eps);
        for (unsigned d = lane; d < dim; d += 32) {
            float v = __bfloat162float(core[off + d]) * inv * __bfloat162float(gamma[d]);
            float gate = __bfloat162float(z[off + d]);
            out[off + d] = __float2bfloat16(v * qwen_silu(gate));
        }
    }
}

static __device__ void d_qwen_q_gate_split(__nv_bfloat16* q, __nv_bfloat16* gate,
    const __nv_bfloat16* packed, const int* active, unsigned heads, unsigned dim,
    unsigned batch, unsigned slice, unsigned nblk) {
    const unsigned cols = heads * dim;
    if ((dim & 7u) == 0) {
        // Eight elements (16 bytes) per thread step; the scalar loop was latency-bound at
        // 4096 x 6144 elements (0.25 ms per op).
        const unsigned n8 = batch * cols / 8u;
        for (unsigned i8 = slice * blockDim.x + threadIdx.x; i8 < n8; i8 += nblk * blockDim.x) {
            const size_t i = (size_t)i8 * 8u;
            if (active && active[i / cols] <= 0) continue;
            const size_t src = (i / dim) * 2 * dim + i % dim;
            *(uint4*)(q + i) = *(const uint4*)(packed + src);
            *(uint4*)(gate + i) = *(const uint4*)(packed + src + dim);
        }
        return;
    }
    for (unsigned i = slice * blockDim.x + threadIdx.x; i < batch * cols; i += nblk * blockDim.x) {
        if (active && active[i / cols] <= 0) continue;
        const size_t src = (size_t)(i / dim) * 2 * dim + i % dim;
        q[i] = packed[src]; gate[i] = packed[src + dim];
    }
}

static __device__ void d_qwen_sigmoid_gate(__nv_bfloat16* out, const __nv_bfloat16* in,
    const __nv_bfloat16* gate, const int* active, unsigned cols, unsigned batch,
    unsigned slice, unsigned nblk) {
    if ((cols & 7u) == 0) {
        // Eight elements (16 bytes) per thread step, same per-element math (0.8 ms -> ~0.1 per
        // op at 4096 x 6144).
        const unsigned n8 = batch * cols / 8u;
        for (unsigned i8 = slice * blockDim.x + threadIdx.x; i8 < n8; i8 += nblk * blockDim.x) {
            const size_t i = (size_t)i8 * 8u;
            if (active && active[i / cols] <= 0) continue;
            const uint4 vi = *(const uint4*)(in + i), vg = *(const uint4*)(gate + i);
            const __nv_bfloat16* pi = (const __nv_bfloat16*)&vi;
            const __nv_bfloat16* pg = (const __nv_bfloat16*)&vg;
            __nv_bfloat16 o[8];
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const float g = __bfloat162float(__float2bfloat16(qwen_sigmoid(__bfloat162float(pg[j]))));
                o[j] = __float2bfloat16(__bfloat162float(pi[j]) * g);
            }
            *(uint4*)(out + i) = *(const uint4*)o;
        }
        return;
    }
    for (unsigned i = slice * blockDim.x + threadIdx.x; i < batch * cols; i += nblk * blockDim.x) {
        if (active && active[i / cols] <= 0) continue;
        float g = __bfloat162float(__float2bfloat16(qwen_sigmoid(__bfloat162float(gate[i]))));
        out[i] = __float2bfloat16(__bfloat162float(in[i]) * g);
    }
}

// One BLOCK per row (block_sum over `part`, the interpreter's shared arena): a 5120-wide row on
// one warp was a serial 160-step load chain, 71 us per instruction on the 27B decode step
// (129 norms = 9 of 38 ms). Every thread reads dim/blockDim elements; `active` is per row, so
// the skip is block-uniform.
// `radd` (t4, optional): the residual add rides the norm. r = bf16(in + radd) is written back
// over `in` (the residual stream, in place as the separate Residual instruction wrote it) and
// normalized: the same rounding as the Residual + QwenRmsNorm pair, one grid-wide dependency
// round fewer (128 pairs per program). Each lane reads back only the elements it wrote.
// `WIDE` (the light entry only): the warp-per-row path stages S chunks per lane and splits a row
// over up to four warps; inside the interpreter objects the staged chunks alone grew the decode
// entry's stack 320 -> 800 B at the 255-register cap, so they keep the one-chunk walk.
template <bool WIDE = false>
static __device__ void d_qwen_rmsnorm(__nv_bfloat16* out, const __nv_bfloat16* in,
    const __nv_bfloat16* gamma, const int* active, unsigned dim, unsigned batch,
    float eps, float offset, unsigned slice, unsigned nblk, float* part,
    const __nv_bfloat16* radd = nullptr) {
    __nv_bfloat16* rout = (__nv_bfloat16*)in;
    if (batch > nblk && (dim & 7u) == 0) {
        // Prefill rows: one warp per row with 16-byte loads (block-per-row ran 4096 rows as 31
        // serial block reductions per block: 0.45 ms per norm). Decode rows keep the block body.
        // S chunks per lane are loaded before their math: with one row per warp (the 1024-row
        // serving chunk on the light grid) the one-chunk walk was a 20-step dependent load chain
        // per pass, 45 us per norm (128 per chunk) for a ~10 us op.
        constexpr unsigned S = WIDE ? 5 : 1;
        const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
        // Warps per row: 4 when the grid holds four per row (the 1024-row serving chunk on the
        // 528-block light grid: one warp per row used 128 of the 528 blocks, 2.5 KB in flight per
        // row, 36-60 us per norm for a ~12 us byte floor), else 2, else 1 — the fat prefill
        // window's 8192-row walk, unchanged. The group's partial sums combine through `part`
        // (block-uniform loop and syncs; a different summation order than one warp per row).
        const unsigned wpr = !WIDE ? 1u
            : batch * 4u <= nblk * warps ? 4u : batch * 2u <= nblk * warps ? 2u : 1u;
        const unsigned groups = warps / wpr, g = warp / wpr, sub = warp - g * wpr;
        const unsigned lstride = 256u * wpr;
        for (unsigned r0 = slice * groups; r0 < batch; r0 += nblk * groups) {
            const unsigned row = r0 + g;
            const bool live = row < batch && !(active && active[row] <= 0);
            const size_t base = (size_t)row * dim;
            float sum = 0.f;
            if (live) for (unsigned d0 = (sub * 32u + lane) * 8u; d0 < dim; d0 += lstride * S) {
                uint4 v[S], a[S];
#pragma unroll
                for (unsigned s = 0; s < S; s++) {
                    const unsigned d = d0 + s * lstride;
                    if (d < dim) {
                        v[s] = *(const uint4*)(in + base + d);
                        if (radd) a[s] = *(const uint4*)(radd + base + d);
                    }
                }
#pragma unroll
                for (unsigned s = 0; s < S; s++) {
                    const unsigned d = d0 + s * lstride;
                    if (d >= dim) continue;
                    __nv_bfloat16* p = (__nv_bfloat16*)&v[s];
                    if (radd) {
                        const __nv_bfloat16* pa = (const __nv_bfloat16*)&a[s];
#pragma unroll
                        for (int j = 0; j < 8; j++)
                            p[j] = __float2bfloat16(__bfloat162float(p[j]) + __bfloat162float(pa[j]));
                        *(uint4*)(rout + base + d) = v[s];
                    }
#pragma unroll
                    for (int j = 0; j < 8; j++) { const float f = __bfloat162float(p[j]); sum += f * f; }
                }
            }
            sum = qwen_warp_sum(sum);
            if (wpr > 1) {
                __syncthreads();
                if (lane == 0) part[warp] = sum;
                __syncthreads();
                sum = 0.f;
                for (unsigned w = 0; w < wpr; w++) sum += part[g * wpr + w];
            }
            if (!live) continue;
            const float inv = rsqrtf(sum / dim + eps);
            for (unsigned d0 = (sub * 32u + lane) * 8u; d0 < dim; d0 += lstride * S) {
                uint4 v[S], gm[S];
#pragma unroll
                for (unsigned s = 0; s < S; s++) {
                    const unsigned d = d0 + s * lstride;
                    if (d < dim) {
                        v[s] = *(const uint4*)(in + base + d);
                        gm[s] = *(const uint4*)(gamma + d);
                    }
                }
#pragma unroll
                for (unsigned s = 0; s < S; s++) {
                    const unsigned d = d0 + s * lstride;
                    if (d >= dim) continue;
                    const __nv_bfloat16* p = (const __nv_bfloat16*)&v[s];
                    const __nv_bfloat16* pg = (const __nv_bfloat16*)&gm[s];
                    __nv_bfloat16 o[8];
#pragma unroll
                    for (int j = 0; j < 8; j++)
                        o[j] = __float2bfloat16(__bfloat162float(p[j]) * inv * (__bfloat162float(pg[j]) + offset));
                    *(uint4*)(out + base + d) = *(const uint4*)o;
                }
            }
        }
        return;
    }
    for (unsigned row = slice; row < batch; row += nblk) {
        if (active && active[row] <= 0) continue;
        const size_t base = (size_t)row * dim;
        float sum = 0.f;
        for (unsigned d = threadIdx.x; d < dim; d += blockDim.x) {
            float x = __bfloat162float(in[base + d]);
            if (radd) {
                const __nv_bfloat16 r = __float2bfloat16(x + __bfloat162float(radd[base + d]));
                rout[base + d] = r;
                x = __bfloat162float(r);
            }
            sum += x * x;
        }
        const float inv = rsqrtf(block_sum(sum, part) / dim + eps);
        for (unsigned d = threadIdx.x; d < dim; d += blockDim.x)
            out[base + d] = __float2bfloat16(__bfloat162float(in[base + d]) * inv *
                (__bfloat162float(gamma[d]) + offset));
    }
}

// Residual + RMSNorm prologue of a decode projection (GEMV / GEMV_GLU arms, i3 == 4): every block
// computes hn = bf16(r * rsqrt(mean(r^2) + eps) * (gamma + offset)) with r = bf16(a + b) (b
// null: r = a) for all rows, stores r to resid_out (null: no store) and reads hn back after the
// block barrier. The same per-element math and rounding as the separate Residual + QwenRmsNorm
// pair the wider rungs run, so a token's numerics do not depend on the rung it lands in
// (normalizing the fp32 sum instead moved the FP32-gate tail by 25% either way: knife-edge
// positions, not precision). The writes are identical across blocks, so no block waits for
// another: the two grid-wide dependency rounds of the separate instructions (~10 us each at this
// grid, 128 norms per step) are gone. `a` must not alias resid_out — another block may still be
// reading `a` (the emitter ping-pongs the residual stream). Row sums are a block (few rows) or
// warp (many rows) reduction. Latency is the whole cost: a warp walking 5120 elements one 16-byte
// chunk per step, twice, cost 160 us per fold at B1 — the block-wide pass runs the row in three
// loads per thread. Rungs above the fold cap (devgen PLOW_QWEN_FOLD_MAX) keep the separate
// instructions.
static __device__ __forceinline__ void qwen_fold_residual8(const __nv_bfloat16* a,
    const __nv_bfloat16* b, size_t at, float (&r)[8]) {
    const uint4 va = *(const uint4*)(a + at);
    const __nv_bfloat16* pa = (const __nv_bfloat16*)&va;
#pragma unroll
    for (int j = 0; j < 8; j++) r[j] = __bfloat162float(pa[j]);
    if (b) {
        const uint4 vb = *(const uint4*)(b + at);
        const __nv_bfloat16* pb = (const __nv_bfloat16*)&vb;
#pragma unroll
        for (int j = 0; j < 8; j++)
            r[j] = __bfloat162float(__float2bfloat16(r[j] + __bfloat162float(pb[j])));
    }
}
static __device__ __forceinline__ void qwen_fold_store8(const float (&r)[8], __nv_bfloat16* out) {
    __nv_bfloat16 o[8];
#pragma unroll
    for (int j = 0; j < 8; j++) o[j] = __float2bfloat16(r[j]);
    *(uint4*)out = *(const uint4*)o;
}
static __device__ __forceinline__ void qwen_fold_norm8(const float (&r)[8],
    const __nv_bfloat16* gamma, size_t d, float inv, float offset, __nv_bfloat16* hn,
    __nv_bfloat16* hn2 = nullptr) {
    const uint4 vg = *(const uint4*)(gamma + d);
    const __nv_bfloat16* pg = (const __nv_bfloat16*)&vg;
    __nv_bfloat16 o[8];
#pragma unroll
    for (int j = 0; j < 8; j++)
        o[j] = __float2bfloat16(r[j] * inv * (__bfloat162float(pg[j]) + offset));
    *(uint4*)(hn + d) = *(const uint4*)o;
    if (hn2) *(uint4*)(hn2 + d) = *(const uint4*)o;
}
// `xs` (shared, rows x dim): the normalized rows also land there, for a projection that reads its
// activations from the arena; `store` false skips the global hn / resid_out writes (one block
// keeps them for the sibling consumers and the residual stream: 132 blocks writing the same
// 20 KB per fold cost more than the walk at B1).
static __device__ void d_qwen_fold_norm(__nv_bfloat16* hn, __nv_bfloat16* resid_out,
    const __nv_bfloat16* a, const __nv_bfloat16* b, const __nv_bfloat16* gamma,
    unsigned rows, unsigned dim, float eps, float offset, float* part,
    __nv_bfloat16* xs = nullptr, bool store = true) {
    if (dim & 7u) { if (threadIdx.x == 0) __trap(); return; }
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    if (!xs) { xs = hn; hn = nullptr; store = true; }
    if (!store) { hn = nullptr; resid_out = nullptr; }
    if (rows < warps) {
        for (unsigned row = 0; row < rows; row++) {
            const size_t base = (size_t)row * dim;
            float sum = 0.f;
            for (unsigned d = threadIdx.x * 8u; d < dim; d += blockDim.x * 8u) {
                float r[8];
                qwen_fold_residual8(a, b, base + d, r);
                if (resid_out) qwen_fold_store8(r, resid_out + base + d);
#pragma unroll
                for (int j = 0; j < 8; j++) sum += r[j] * r[j];
            }
            const float inv = rsqrtf(block_sum(sum, part) / dim + eps);
            for (unsigned d = threadIdx.x * 8u; d < dim; d += blockDim.x * 8u) {
                float r[8];
                qwen_fold_residual8(a, b, base + d, r);
                qwen_fold_norm8(r, gamma, d, inv, offset, xs + base, hn ? hn + base : nullptr);
            }
        }
        return;
    }
    for (unsigned row = warp; row < rows; row += warps) {
        const size_t base = (size_t)row * dim;
        float sum = 0.f;
        for (unsigned d = lane * 8u; d < dim; d += 256u) {
            float r[8];
            qwen_fold_residual8(a, b, base + d, r);
            if (resid_out) qwen_fold_store8(r, resid_out + base + d);
#pragma unroll
            for (int j = 0; j < 8; j++) sum += r[j] * r[j];
        }
        const float inv = rsqrtf(qwen_warp_sum(sum) / dim + eps);
        for (unsigned d = lane * 8u; d < dim; d += 256u) {
            float r[8];
            qwen_fold_residual8(a, b, base + d, r);
            qwen_fold_norm8(r, gamma, d, inv, offset, xs + base, hn ? hn + base : nullptr);
        }
    }
}

#ifndef PLOW_NV_KV_ILV
#define PLOW_NV_KV_ILV 0  /* op_attention.cuh: head-interleaved KV rows */
#endif
static __device__ void d_qwen_headnorm_rope(__nv_bfloat16* out, const __nv_bfloat16* in,
    const __nv_bfloat16* gamma, const float* cos, const float* sin, const int* positions,
    const int* active, unsigned heads, unsigned dim, unsigned rotary, unsigned batch,
    unsigned context, unsigned normalize, float eps, float offset, unsigned slice, unsigned nblk,
    unsigned prefill = 0, const int* pfslot = nullptr) {
    if (dim != 256 || (rotary != 0 && rotary != 64)) { if (threadIdx.x == 0) __trap(); return; }
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    for (unsigned row = slice * warps + warp; row < batch * heads; row += nblk * warps) {
        const unsigned slot = row / heads, head = row % heads;
        if (active && active[slot] <= 0) continue;
        // Packed prefill: the row's KV slot from the per-row slot table (unshifted KV tensor).
        unsigned kvslot = prefill ? 0u : slot;
        if (pfslot) { const int ps = pfslot[slot]; if (ps < 0) continue; kvslot = (unsigned)ps; }
        const int pos = positions ? positions[slot] : 0;
        if (pos < 0 || (context && (unsigned)pos >= context)) { if (lane == 0) __trap(); continue; }
        float x[8], sum = 0.f;
#pragma unroll
        for (unsigned j = 0; j < 8; j++) {
            x[j] = __bfloat162float(in[(size_t)row * dim + lane + j * 32]); sum += x[j] * x[j];
        }
        if (normalize) {
            const float inv = rsqrtf(qwen_warp_sum(sum) / dim + eps);
#pragma unroll
            for (unsigned j = 0; j < 8; j++) {
                float g = gamma ? __bfloat162float(gamma[lane + j * 32]) + offset : 1.f;
                x[j] = __bfloat162float(__float2bfloat16(x[j] * inv * g));
            }
        }
        if (rotary) {
            // vLLM stores its Qwen rotary cache in the model BF16 dtype.
            float c = __bfloat162float(__float2bfloat16(cos[(size_t)pos * 32 + lane]));
            float s = __bfloat162float(__float2bfloat16(sin[(size_t)pos * 32 + lane]));
            float lo = x[0], hi = x[1];
            x[0] = lo * c - hi * s; x[1] = hi * c + lo * s;
        }
#if PLOW_NV_KV_ILV
        size_t dest = context ? (((size_t)kvslot * context + pos) * heads + head) * dim
                              : (size_t)row * dim;
#else
        size_t dest = context ? ((size_t)kvslot * heads + head) * context * dim + (size_t)pos * dim
                              : (size_t)row * dim;
#endif
#pragma unroll
        for (unsigned j = 0; j < 8; j++) out[dest + lane + j * 32] = __float2bfloat16(x[j]);
    }
}

// Channel `c` of a sequence shorter than width-1 rows, walked serially.
static __device__ __forceinline__ void d_qwen_gdn_conv_serial_channel(__nv_bfloat16* out,
    const __nv_bfloat16* in, const __nv_bfloat16* weight, __nv_bfloat16* history,
    unsigned channels, unsigned width, unsigned tokens, unsigned c) {
    const __nv_bfloat16* w = weight + (size_t)c * width;
    __nv_bfloat16* h = history + (size_t)c * (width - 1);
    for (unsigned t = 0; t < tokens; t++) {
        float sum = 0.f;
        for (unsigned j = 0; j < width; j++) {
            int source = (int)t + (int)j - (int)(width - 1);
            float x = source < 0 ? __bfloat162float(h[t + j])
                : __bfloat162float(in[(size_t)source * channels + c]);
            sum = fmaf(x, __bfloat162float(w[j]), sum);
        }
        out[(size_t)t * channels + c] = __float2bfloat16(qwen_silu(sum));
    }
    for (unsigned j = 0; j + 1 < width; j++) {
        int source = (int)tokens + (int)j - (int)(width - 1);
        h[j] = source < 0 ? h[tokens + j] : in[(size_t)source * channels + c];
    }
}

// Rows per (8-channel group, tile) item of the width-4 walk: the smallest tile that fills the
// entry's resident threads (RESB blocks per SM, %nsmid SMs) a whole number of rounds. An item
// costs its rows, so a part-filled last round costs a full one: 16-row tiles at 1024 rows x
// 1280 groups were 81920 items on the light entry's 67584 threads, a second round 21% full.
// Standalone (convbench2, H100): 1024 rows 29.1 -> 25.4 us with 20-row tiles, 512 rows 20.8 ->
// 17.6 with 10, 4096 rows 87.7 -> 78.6 with 40 (two rounds; one round of 78 would exceed the
// register window). Multiples of DEPTH, at most 64 rows.
template <unsigned DEPTH, unsigned RESB>
static __device__ __forceinline__ unsigned qwen_conv4_tile_rows(unsigned rows, unsigned cg) {
    unsigned nsm;
    asm("mov.u32 %0, %%nsmid;" : "=r"(nsm));
    const unsigned long long resident = (unsigned long long)nsm * RESB * blockDim.x;
    const unsigned long long row_items = (unsigned long long)rows * cg;
    for (unsigned k = 1;; k++) {
        unsigned vt = (unsigned)((row_items + k * resident - 1) / (k * resident));
        vt = (vt + DEPTH - 1) / DEPTH * DEPTH;
        if (vt < DEPTH) vt = DEPTH;
        if (vt <= 64) return vt;
    }
}

// One (8-channel group, `vt`-row tile) item of the width-4 (Qwen3.5) fast path: eight channels
// (16 B) per thread. The 2-byte per-channel tile walk was latency-bound: 12.8 / 50.6 ms at
// 1024 / 4096 rows against a 0.6 / 2.4 ms byte floor. Only the tile-0 thread reads the history
// (sources before the chunk) and, once its tile is done, rewrites it from the chunk's last
// width-1 rows; every other tile's taps lie inside the chunk. The window streams: three live
// rows plus the next DEPTH loaded together, unpacked by bit shifts so no vector takes an
// address. Rolled groups of DEPTH rows with the next group's loads in flight: unrolled, the
// 128 sigmoids of a 16-row tile were ~4000 straight-line instructions (64 KB) that every warp
// streamed once, and ncu put 46% of the light conv's warp stalls on instruction fetch; rolled,
// the row loads showed (long_scoreboard 23%). 1024-row site, light entry: 61 -> 43 (rolled, 4
// rows) -> 39 us (2 rows). Same fmaf chain and sigmoid per output as the serial walk below.
template <unsigned DEPTH>
static __device__ __forceinline__ void d_qwen_gdn_conv4_tile(__nv_bfloat16* out,
    const __nv_bfloat16* in, const __nv_bfloat16* weight, __nv_bfloat16* history,
    unsigned channels, unsigned tokens, unsigned c0, unsigned tile, unsigned vt) {
    constexpr unsigned PD = DEPTH;
    const unsigned t0 = tile * vt, t1 = min(t0 + vt, tokens), nt = t1 - t0;
    float wf[4][8];
#pragma unroll
    for (int ch = 0; ch < 8; ch++) {
        const __nv_bfloat16* w = weight + (size_t)(c0 + ch) * 4u;
#pragma unroll
        for (int j = 0; j < 4; j++) wf[j][ch] = __bfloat162float(w[j]);
    }
    float r0[8], r1[8], r2[8];
    if (tile == 0) {
#pragma unroll
        for (int ch = 0; ch < 8; ch++) {
            const __nv_bfloat16* h = history + (size_t)(c0 + ch) * 3u;
            r0[ch] = __bfloat162float(h[0]);
            r1[ch] = __bfloat162float(h[1]);
            r2[ch] = __bfloat162float(h[2]);
        }
    } else {
        qwen_bf_unpack8(*(const uint4*)(in + (size_t)(t0 - 3) * channels + c0), r0);
        qwen_bf_unpack8(*(const uint4*)(in + (size_t)(t0 - 2) * channels + c0), r1);
        qwen_bf_unpack8(*(const uint4*)(in + (size_t)(t0 - 1) * channels + c0), r2);
    }
    uint4 nx[PD];
#pragma unroll
    for (unsigned u = 0; u < PD; u++)
        nx[u] = u < nt ? *(const uint4*)(in + (size_t)(t0 + u) * channels + c0)
                       : make_uint4(0u, 0u, 0u, 0u);
#pragma unroll 1
    for (unsigned t = 0; t < nt; t += PD) {
        uint4 nn[PD];
#pragma unroll
        for (unsigned u = 0; u < PD; u++)
            nn[u] = t + PD + u < nt ? *(const uint4*)(in + (size_t)(t0 + t + PD + u) * channels + c0)
                                    : make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
        for (unsigned u = 0; u < PD; u++) {
            float x[8], o[8];
            qwen_bf_unpack8(nx[u], x);
#pragma unroll
            for (int ch = 0; ch < 8; ch++) {
                float sum = fmaf(r0[ch], wf[0][ch], 0.f);
                sum = fmaf(r1[ch], wf[1][ch], sum);
                sum = fmaf(r2[ch], wf[2][ch], sum);
                sum = fmaf(x[ch], wf[3][ch], sum);
                o[ch] = qwen_silu(sum);
                r0[ch] = r1[ch]; r1[ch] = r2[ch]; r2[ch] = x[ch];
            }
            if (t + u < nt) *(uint4*)(out + (size_t)(t0 + t + u) * channels + c0) = qwen_bf_pack8(o);
        }
#pragma unroll
        for (unsigned u = 0; u < PD; u++) nx[u] = nn[u];
    }
    if (tile == 0) {
#pragma unroll
        for (int j = 0; j < 3; j++)
            for (int ch = 0; ch < 8; ch++)
                history[(size_t)(c0 + ch) * 3u + j] = in[(size_t)(tokens + j - 3) * channels + c0 + ch];
    }
}

// One sequence of `tokens` rows: taps before row 0 come from `history` (this slot's last
// width-1 rows), which is rewritten from the sequence's last rows.
// `DEPTH`: rows of the width-4 fast path per loop group, with the next group's rows loaded
// during this one (2 x DEPTH in flight). 4 on the prefill interpreter; 2 on the 128-register
// light entry (per 1024-row site: 4 -> 43 us, 2 -> 39, 1 -> 43 before the double buffering).
// `RESB`: the entry's resident blocks per SM (its launch bounds), which size the tiles.
template <bool PF, unsigned DEPTH = 4, unsigned RESB = 1>
static __device__ void d_qwen_gdn_conv_prefill_seq(__nv_bfloat16* out, const __nv_bfloat16* in,
    const __nv_bfloat16* weight, __nv_bfloat16* history, unsigned channels,
    unsigned width, unsigned tokens, unsigned slice, unsigned nblk) {
    // Tiles of TILE tokens per (channel, tile) thread: a serial per-channel walk left 100 of 132
    // blocks idle and cost 123 ms for 4096 rows (48 layers). Only the tile-0 thread reads the
    // history (sources before the chunk) and, once its tile is done, rewrites it from the chunk's
    // last width-1 rows; every other tile's taps lie inside the chunk. Same fmaf chain per output
    // as the serial walk below, which chunks shorter than width-1 rows keep.
    if constexpr (PF) {
    // Prefill instantiations only: the staged rows cost the decode entry (already at the
    // 255-register cap) stack for an op its programs never run.
    if (width == 4 && (channels & 7u) == 0 && tokens >= 3) {
        const unsigned cg = channels >> 3, vt = qwen_conv4_tile_rows<DEPTH, RESB>(tokens, cg);
        const unsigned ntiles = (tokens + vt - 1) / vt;
        for (unsigned i = slice * blockDim.x + threadIdx.x; i < cg * ntiles; i += nblk * blockDim.x)
            d_qwen_gdn_conv4_tile<DEPTH>(out, in, weight, history, channels, tokens, (i % cg) * 8u, i / cg, vt);
        return;
    }
    }
    constexpr unsigned TILE = 64;
    if (tokens + 1 >= width) {
        const unsigned ntiles = (tokens + TILE - 1) / TILE;
        for (unsigned i = slice * blockDim.x + threadIdx.x; i < channels * ntiles;
             i += nblk * blockDim.x) {
            const unsigned c = i % channels, tile = i / channels;
            const unsigned t0 = tile * TILE, t1 = min(t0 + TILE, tokens);
            const __nv_bfloat16* w = weight + (size_t)c * width;
            __nv_bfloat16* h = history + (size_t)c * (width - 1);
            float wf[8], win[8];
            for (unsigned j = 0; j < width; j++) wf[j] = __bfloat162float(w[j]);
            for (unsigned j = 0; j + 1 < width; j++) {
                const int source = (int)t0 + (int)j - (int)(width - 1);
                win[j] = source < 0 ? __bfloat162float(h[t0 + j])
                    : __bfloat162float(in[(size_t)source * channels + c]);
            }
            for (unsigned t = t0; t < t1; t++) {
                const float x = __bfloat162float(in[(size_t)t * channels + c]);
                float sum = 0.f;
                for (unsigned j = 0; j + 1 < width; j++) sum = fmaf(win[j], wf[j], sum);
                sum = fmaf(x, wf[width - 1], sum);
                out[(size_t)t * channels + c] = __float2bfloat16(qwen_silu(sum));
                for (unsigned j = 0; j + 2 < width; j++) win[j] = win[j + 1];
                win[width - 2] = x;
            }
            if (tile == 0)
                for (unsigned j = 0; j + 1 < width; j++)
                    h[j] = in[(size_t)(tokens + j - (width - 1)) * channels + c];
        }
        return;
    }
    for (unsigned c = slice * blockDim.x + threadIdx.x; c < channels; c += nblk * blockDim.x)
        d_qwen_gdn_conv_serial_channel(out, in, weight, history, channels, width, tokens, c);
}

// A request shorter than width-1 rows (a rider's one-row decode request) for one 8-channel
// group: every tap of every row comes from the history or the request's own rows, which the
// register window holds, and the history is rewritten from the window. The 8 channels' 3-row
// history is 48 contiguous bytes (three 16-B vectors). Same fmaf chain and sigmoid per output
// as the serial walk; the window's floats are the bf16 values, so the rewrite is exact.
static __device__ __forceinline__ void d_qwen_gdn_conv4_short(__nv_bfloat16* out,
    const __nv_bfloat16* in, const __nv_bfloat16* weight, __nv_bfloat16* history,
    unsigned channels, unsigned tokens, unsigned c0) {
    float wf[4][8];
#pragma unroll
    for (int ch = 0; ch < 8; ch++) {
        const __nv_bfloat16* w = weight + (size_t)(c0 + ch) * 4u;
#pragma unroll
        for (int j = 0; j < 4; j++) wf[j][ch] = __bfloat162float(w[j]);
    }
    uint4* hv = (uint4*)(history + (size_t)c0 * 3u);
    const uint4 h0 = hv[0], h1 = hv[1], h2 = hv[2];
    // [ch0: r0 r1 r2][ch1: r0 r1 r2]... across the three vectors.
    const unsigned hw[12] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w, h2.x, h2.y, h2.z, h2.w};
    float r0[8], r1[8], r2[8];
#pragma unroll
    for (int ch = 0; ch < 8; ch++) {
        const unsigned e = 3u * ch;
        r0[ch] = (e & 1u) ? qwen_bf_hi(hw[e >> 1]) : qwen_bf_lo(hw[e >> 1]);
        r1[ch] = ((e + 1u) & 1u) ? qwen_bf_hi(hw[(e + 1u) >> 1]) : qwen_bf_lo(hw[(e + 1u) >> 1]);
        r2[ch] = ((e + 2u) & 1u) ? qwen_bf_hi(hw[(e + 2u) >> 1]) : qwen_bf_lo(hw[(e + 2u) >> 1]);
    }
    for (unsigned t = 0; t < tokens; t++) {
        float x[8], o[8];
        qwen_bf_unpack8(*(const uint4*)(in + (size_t)t * channels + c0), x);
#pragma unroll
        for (int ch = 0; ch < 8; ch++) {
            float sum = fmaf(r0[ch], wf[0][ch], 0.f);
            sum = fmaf(r1[ch], wf[1][ch], sum);
            sum = fmaf(r2[ch], wf[2][ch], sum);
            sum = fmaf(x[ch], wf[3][ch], sum);
            o[ch] = qwen_silu(sum);
            r0[ch] = r1[ch]; r1[ch] = r2[ch]; r2[ch] = x[ch];
        }
        *(uint4*)(out + (size_t)t * channels + c0) = qwen_bf_pack8(o);
    }
    float hf[24];
#pragma unroll
    for (int ch = 0; ch < 8; ch++) { hf[3 * ch] = r0[ch]; hf[3 * ch + 1] = r1[ch]; hf[3 * ch + 2] = r2[ch]; }
    hv[0] = make_uint4(qwen_bf_pack(hf[0], hf[1]), qwen_bf_pack(hf[2], hf[3]), qwen_bf_pack(hf[4], hf[5]), qwen_bf_pack(hf[6], hf[7]));
    hv[1] = make_uint4(qwen_bf_pack(hf[8], hf[9]), qwen_bf_pack(hf[10], hf[11]), qwen_bf_pack(hf[12], hf[13]), qwen_bf_pack(hf[14], hf[15]));
    hv[2] = make_uint4(qwen_bf_pack(hf[16], hf[17]), qwen_bf_pack(hf[18], hf[19]), qwen_bf_pack(hf[20], hf[21]), qwen_bf_pack(hf[22], hf[23]));
}

// Exclusive prefix sums of two per-entry arrays in shared memory (n <= 32 x blockDim; a[n] and
// b[n] receive the totals), by warp 0: lane-strided segments, a shuffle scan across the lanes.
static __device__ __forceinline__ void qwen_pack_scan2(unsigned* a, unsigned* b, unsigned n) {
    if (threadIdx.x < 32) {
        const unsigned lane = threadIdx.x, per = (n + 31u) >> 5, r0 = lane * per, r1 = min(r0 + per, n);
        unsigned ta = 0, tb = 0;
        for (unsigned r = r0; r < r1; r++) { ta += a[r]; tb += b[r]; }
        unsigned ia = ta, ib = tb;
#pragma unroll
        for (unsigned o = 1; o < 32; o <<= 1) {
            const unsigned ua = __shfl_up_sync(0xffffffffu, ia, o), ub = __shfl_up_sync(0xffffffffu, ib, o);
            if (lane >= o) { ia += ua; ib += ub; }
        }
        unsigned ea = ia - ta, eb = ib - tb;
        for (unsigned r = r0; r < r1; r++) {
            const unsigned va = a[r], vb = b[r];
            a[r] = ea; b[r] = eb;
            ea += va; eb += vb;
        }
        if (lane == 31) { a[n] = ia; b[n] = ib; }
    }
}

// First entry r in [0, n) whose exclusive prefix end `p[r + 1]` exceeds `x` (p non-decreasing).
static __device__ __forceinline__ unsigned qwen_pack_find(const unsigned* p, unsigned n, unsigned x) {
    unsigned lo = 0, hi = n;
    while (lo < hi) {
        const unsigned mid = (lo + hi) >> 1;
        if (p[mid + 1] > x) hi = mid; else lo = mid + 1;
    }
    return lo;
}

// `rows` > `tokens`: a padded chunk (plowrt patches tokens = real rows, rows = bucket); the pad
// rows' outputs are zeroed so the layers after the conv see finite activations. `req` (packed
// prefill): the request table [n, (row0, len, slot, end) x n]; each request is its own sequence
// with its slot's history (history is the unshifted per-slot tensor), tokens = the real rows.
// The whole block calls this (its barriers).
template <bool PF, unsigned DEPTH = 4, unsigned RESB = 1>
static __device__ void d_qwen_gdn_conv_prefill(__nv_bfloat16* out, const __nv_bfloat16* in,
    const __nv_bfloat16* weight, __nv_bfloat16* history, unsigned channels,
    unsigned width, unsigned tokens, unsigned rows, unsigned slice, unsigned nblk,
    const int* req = nullptr) {
    if (width < 2 || width > 8 || !tokens) { if (threadIdx.x == 0) __trap(); return; }
    if (rows > tokens) {
        const size_t pad0 = (size_t)tokens * channels, pad1 = (size_t)rows * channels;
        for (size_t i = pad0 + slice * blockDim.x + threadIdx.x; i < pad1; i += (size_t)nblk * blockDim.x)
            out[i] = __float2bfloat16(0.f);
    }
    if (!req) {
        d_qwen_gdn_conv_prefill_seq<PF, DEPTH, RESB>(out, in, weight, history, channels, width, tokens, slice, nblk);
        return;
    }
    const unsigned n = (unsigned)req[0];
    if constexpr (PF) {
    if (width == 4 && (channels & 7u) == 0) {
        // All requests' items in one pass: the long requests' (tile, channel group) items, tiles
        // sized by the rows of all of them with a rider counted as two rows (its item loads the
        // history and the weights for one row), then the short requests' (request, channel
        // group) items, which run alongside the tiles rather than after them. A pass per request
        // left the grid ~30% busy on the 512 + 501-row ticks of 4K c32 serving (66 us per conv
        // vs 39 for one 1013-row request), and a per-channel rider pass after the tiles kept the
        // serving conv at 58 us when the same rows without riders took 33. The table is reduced
        // once per block into shared prefix sums (tiles and short requests per entry) and each
        // item finds its request by a binary search over them: a per-thread walk of the table
        // per item (a dependent load and a division per entry, every thread of the grid, ~3.5
        // walks each) was pure issue time, 32 riders alone 12-19 us. Same per-tile walk.
        constexpr unsigned PMAX = 256;
        __shared__ unsigned s_tile[PMAX + 1], s_short[PMAX + 1];
        if (n > PMAX) { if (threadIdx.x == 0) __trap(); return; }
        const unsigned cg = channels >> 3;
        for (unsigned r = threadIdx.x; r < n; r += blockDim.x) {
            const unsigned row0 = (unsigned)req[1 + 4 * r], len = (unsigned)req[2 + 4 * r];
            if (!len || row0 + len > tokens) __trap();
            const bool lng = len + 1 >= width;
            s_tile[r] = lng ? len : 0u;
            s_short[r] = lng ? 0u : 1u;
        }
        __syncthreads();
        qwen_pack_scan2(s_tile, s_short, n);
        __syncthreads();
        const unsigned long_rows = s_tile[n], n_short = s_short[n];
        const unsigned vt = qwen_conv4_tile_rows<DEPTH, RESB>(long_rows + 2u * n_short, cg);
        __syncthreads();
        for (unsigned r = threadIdx.x; r < n; r += blockDim.x) {
            const unsigned len = (unsigned)req[2 + 4 * r];
            const bool lng = len + 1 >= width;
            s_tile[r] = lng ? (len + vt - 1) / vt : 0u;
            s_short[r] = lng ? 0u : 1u;
        }
        __syncthreads();
        qwen_pack_scan2(s_tile, s_short, n);
        __syncthreads();
        const unsigned long_items = cg * s_tile[n], items = long_items + cg * n_short;
        for (unsigned i = slice * blockDim.x + threadIdx.x; i < items; i += nblk * blockDim.x) {
            const unsigned c0 = (i % cg) * 8u;
            if (i < long_items) {
                const unsigned tile = i / cg, r = qwen_pack_find(s_tile, n, tile);
                const unsigned row0 = (unsigned)req[1 + 4 * r], len = (unsigned)req[2 + 4 * r];
                const unsigned slot = (unsigned)req[3 + 4 * r];
                d_qwen_gdn_conv4_tile<DEPTH>(out + (size_t)row0 * channels, in + (size_t)row0 * channels,
                    weight, history + (size_t)slot * channels * 3u, channels, len, c0, tile - s_tile[r], vt);
            } else {
                const unsigned r = qwen_pack_find(s_short, n, (i - long_items) / cg);
                const unsigned row0 = (unsigned)req[1 + 4 * r], len = (unsigned)req[2 + 4 * r];
                const unsigned slot = (unsigned)req[3 + 4 * r];
                d_qwen_gdn_conv4_short(out + (size_t)row0 * channels, in + (size_t)row0 * channels,
                    weight, history + (size_t)slot * channels * 3u, channels, len, c0);
            }
        }
        return;
    }
    }
    for (unsigned r = 0; r < n; r++) {
        const unsigned row0 = (unsigned)req[1 + 4 * r], len = (unsigned)req[2 + 4 * r];
        if (!len || row0 + len > tokens) { if (threadIdx.x == 0) __trap(); return; }
    }
    for (unsigned r = 0; r < n; r++) {
        const unsigned row0 = (unsigned)req[1 + 4 * r], len = (unsigned)req[2 + 4 * r];
        const unsigned slot = (unsigned)req[3 + 4 * r];
        if (len + 1 < width) continue;
        d_qwen_gdn_conv_prefill_seq<PF, DEPTH, RESB>(out + (size_t)row0 * channels, in + (size_t)row0 * channels, weight,
            history + (size_t)slot * channels * (width - 1), channels, width, len, slice, nblk);
    }
    // Requests shorter than width-1 rows (the riders' one-row decode requests) in one flat pass:
    // one serial pass per request cost a dependent-load round trip each, 25 riders ~50 us on a
    // ~37 us 1024-row conv (Qwen3.8-27B 4K c32 serving). Same per-channel walk.
    for (unsigned i = slice * blockDim.x + threadIdx.x; i < n * channels; i += nblk * blockDim.x) {
        const unsigned r = i / channels, c = i - r * channels;
        const unsigned row0 = (unsigned)req[1 + 4 * r], len = (unsigned)req[2 + 4 * r];
        if (len + 1 >= width) continue;
        const unsigned slot = (unsigned)req[3 + 4 * r];
        d_qwen_gdn_conv_serial_channel(out + (size_t)row0 * channels, in + (size_t)row0 * channels, weight,
            history + (size_t)slot * channels * (width - 1), channels, width, len, c);
    }
}

template <bool PF>
static __device__ void d_qwen_gdn_qkv_prep(__nv_bfloat16* q, __nv_bfloat16* k,
    __nv_bfloat16* v, const __nv_bfloat16* packed, unsigned hk, unsigned hv,
    unsigned kd, unsigned vd, unsigned tokens, float eps, unsigned slice, unsigned nblk) {
    if (kd != 128 || !hk || !hv || !vd) { if (threadIdx.x == 0) __trap(); return; }
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    const unsigned channels = 2 * hk * kd + hv * vd;
    if constexpr (PF) {
    if ((vd & 7u) == 0) {
        // Prefill objects: eight-byte loads (lane L holds elements 4L..4L+3 of the q and k rows
        // of one (token, head)), four rows per warp step with the eight loads in flight, and the
        // v copy in 16-byte vectors. The 2-byte lane-strided walk with a per-element div/mod copy
        // ran at 12% of the byte floor (20.7 ms per 96 preps at 4096 rows, floor 2.4). Same
        // per-element math and the generic loop's accumulation order (qwen_strided_sq4):
        // bit-identical.
        const unsigned total = tokens * hk;
        constexpr unsigned R = 4;
        for (unsigned row0 = (slice * warps + warp) * R; row0 < total; row0 += nblk * warps * R) {
            uint2 q2[R], k2[R];
#pragma unroll
            for (unsigned r = 0; r < R; r++) {
                const unsigned row = row0 + r, t = row / hk, head = row - t * hk;
                const __nv_bfloat16* src = packed + (size_t)t * channels + head * kd + lane * 4u;
                q2[r] = row < total ? *(const uint2*)src : make_uint2(0u, 0u);
                k2[r] = row < total ? *(const uint2*)(src + hk * kd) : make_uint2(0u, 0u);
            }
#pragma unroll
            for (unsigned r = 0; r < R; r++) {
                const unsigned row = row0 + r;
                float qf[4], kf[4];
                qwen_bf_unpack4(q2[r], qf);
                qwen_bf_unpack4(k2[r], kf);
                const float qinv = rsqrtf(qwen_warp_sum(qwen_strided_sq4_pk(q2[r], lane)) + eps);
                const float kinv = rsqrtf(qwen_warp_sum(qwen_strided_sq4_pk(k2[r], lane)) + eps);
#pragma unroll
                for (int j = 0; j < 4; j++) { qf[j] *= qinv; kf[j] *= kinv; }
                if (row < total) {
                    *(uint2*)(q + (size_t)row * kd + lane * 4u) = qwen_bf_pack4(qf);
                    *(uint2*)(k + (size_t)row * kd + lane * 4u) = qwen_bf_pack4(kf);
                }
            }
        }
        const unsigned vrow8 = (hv * vd) >> 3, n8 = tokens * vrow8, stride = nblk * blockDim.x;
        const __nv_bfloat16* vsrc = packed + 2u * hk * kd;
        for (unsigned i = slice * blockDim.x + threadIdx.x; i < n8; i += 4u * stride) {
            uint4 a[4];
#pragma unroll
            for (unsigned u = 0; u < 4; u++) {
                const unsigned ii = i + u * stride, t = ii / vrow8, c = ii - t * vrow8;
                if (ii < n8) a[u] = *(const uint4*)(vsrc + (size_t)t * channels + (size_t)c * 8u);
            }
#pragma unroll
            for (unsigned u = 0; u < 4; u++) {
                const unsigned ii = i + u * stride;
                if (ii < n8) *(uint4*)(v + (size_t)ii * 8u) = a[u];
            }
        }
        return;
    }
    }
    for (unsigned row = slice * warps + warp; row < tokens * hk; row += nblk * warps) {
        const unsigned t = row / hk, head = row % hk;
        float qv[4], kv[4], qq = 0.f, kk = 0.f;
        for (unsigned j = 0; j < 4; j++) {
            unsigned d = lane + 32 * j;
            qv[j] = __bfloat162float(packed[(size_t)t * channels + head * kd + d]);
            kv[j] = __bfloat162float(packed[(size_t)t * channels + hk * kd + head * kd + d]);
            qq += qv[j] * qv[j]; kk += kv[j] * kv[j];
        }
        float qinv = rsqrtf(qwen_warp_sum(qq) + eps), kinv = rsqrtf(qwen_warp_sum(kk) + eps);
        for (unsigned j = 0; j < 4; j++) {
            size_t dest = (size_t)row * kd + lane + 32 * j;
            q[dest] = __float2bfloat16(qv[j] * qinv);
            k[dest] = __float2bfloat16(kv[j] * kinv);
        }
    }
    for (unsigned i = slice * blockDim.x + threadIdx.x; i < tokens * hv * vd; i += nblk * blockDim.x)
        v[i] = packed[(size_t)(i / (hv * vd)) * channels + 2 * hk * kd + i % (hv * vd)];
}

static __device__ void d_qwen_gdn_gate_prep(float* alpha, float* beta,
    const __nv_bfloat16* a, const __nv_bfloat16* b, const __nv_bfloat16* alog,
    const __nv_bfloat16* bias, unsigned heads, unsigned tokens, unsigned slice, unsigned nblk) {
    for (unsigned i = slice * blockDim.x + threadIdx.x; i < tokens * heads; i += nblk * blockDim.x) {
        const unsigned head = i % heads;
        float dt = __bfloat162float(a[i]) + __bfloat162float(bias[head]);
        float sp = dt > 20.f ? dt : log1pf(expf(dt));
        alpha[i] = expf(-expf(__bfloat162float(alog[head])) * sp);
        beta[i] = __bfloat162float(__float2bfloat16(1.f / (1.f + expf(-__bfloat162float(b[i])))));
    }
}
