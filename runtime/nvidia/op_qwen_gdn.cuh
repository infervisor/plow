#pragma once
#include <cuda_bf16.h>
#include <cmath>

static __device__ __forceinline__ float qwen_warp_sum(float v) {
#pragma unroll
    for (int d = 16; d; d >>= 1) v += __shfl_down_sync(0xffffffffu, v, d);
    return __shfl_sync(0xffffffffu, v, 0);
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

static __device__ void d_qwen_gdn_step(__nv_bfloat16* out, const __nv_bfloat16* qkv,
    const __nv_bfloat16* a, const __nv_bfloat16* b, const void* a_log,
    const __nv_bfloat16* dt_bias, float* state, const int* active,
    unsigned hk, unsigned hv, unsigned kdim, unsigned vdim, unsigned batch,
    float scale, float eps, unsigned alog_f32, unsigned slice, unsigned nblk) {
    if (!hk || !hv || hv % hk || kdim != 128 || !vdim) {
        if (threadIdx.x == 0) __trap(); return;
    }
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const unsigned warps = blockDim.x >> 5;
    const unsigned packed = 2 * hk * kdim + hv * vdim;
    // A warp owns one V row of the V-first state, so no state writers overlap.
    for (unsigned row = slice * warps + warp; row < batch * hv * vdim; row += nblk * warps) {
        const unsigned slot = row / (hv * vdim);
        if (active && active[slot] <= 0) continue;
        const unsigned head = row / vdim % hv, vcol = row % vdim, qhead = head / (hv / hk);
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
            h[j] = state[(size_t)row * kdim + lane + 32 * j] * decay;
            projection += h[j] * k[j];
        }
        const float value = __bfloat162float(x[2 * hk * kdim + head * vdim + vcol]);
        const float delta = (value - qwen_warp_sum(projection)) * beta;
        float result = 0.f;
#pragma unroll
        for (unsigned j = 0; j < 4; j++) {
            h[j] = fmaf(delta, k[j], h[j]);
            state[(size_t)row * kdim + lane + 32 * j] = h[j];
            result += h[j] * q[j];
        }
        result = qwen_warp_sum(result);
        if (lane == 0) out[row] = __float2bfloat16(result);
    }
}

static __device__ void d_qwen_gated_norm(__nv_bfloat16* out, const __nv_bfloat16* core,
    const __nv_bfloat16* z, const __nv_bfloat16* gamma, const int* active,
    unsigned heads, unsigned dim, unsigned batch, float eps, unsigned slice, unsigned nblk) {
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
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
            out[off + d] = __float2bfloat16(v * (gate / (1.f + expf(-gate))));
        }
    }
}

static __device__ void d_qwen_q_gate_split(__nv_bfloat16* q, __nv_bfloat16* gate,
    const __nv_bfloat16* packed, const int* active, unsigned heads, unsigned dim,
    unsigned batch, unsigned slice, unsigned nblk) {
    const unsigned cols = heads * dim;
    for (unsigned i = slice * blockDim.x + threadIdx.x; i < batch * cols; i += nblk * blockDim.x) {
        if (active && active[i / cols] <= 0) continue;
        const size_t src = (size_t)(i / dim) * 2 * dim + i % dim;
        q[i] = packed[src]; gate[i] = packed[src + dim];
    }
}

static __device__ void d_qwen_sigmoid_gate(__nv_bfloat16* out, const __nv_bfloat16* in,
    const __nv_bfloat16* gate, const int* active, unsigned cols, unsigned batch,
    unsigned slice, unsigned nblk) {
    for (unsigned i = slice * blockDim.x + threadIdx.x; i < batch * cols; i += nblk * blockDim.x) {
        if (active && active[i / cols] <= 0) continue;
        float g = __bfloat162float(__float2bfloat16(1.f / (1.f + expf(-__bfloat162float(gate[i])))));
        out[i] = __float2bfloat16(__bfloat162float(in[i]) * g);
    }
}

static __device__ void d_qwen_rmsnorm(__nv_bfloat16* out, const __nv_bfloat16* in,
    const __nv_bfloat16* gamma, const int* active, unsigned dim, unsigned batch,
    float eps, float offset, unsigned slice, unsigned nblk) {
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    for (unsigned row = slice * warps + warp; row < batch; row += nblk * warps) {
        if (active && active[row] <= 0) continue;
        size_t base = (size_t)row * dim;
        float sum = 0.f;
        for (unsigned d = lane; d < dim; d += 32) {
            float x = __bfloat162float(in[base + d]); sum += x * x;
        }
        float inv = rsqrtf(qwen_warp_sum(sum) / dim + eps);
        for (unsigned d = lane; d < dim; d += 32)
            out[base + d] = __float2bfloat16(__bfloat162float(in[base + d]) * inv *
                (__bfloat162float(gamma[d]) + offset));
    }
}

static __device__ void d_qwen_headnorm_rope(__nv_bfloat16* out, const __nv_bfloat16* in,
    const __nv_bfloat16* gamma, const float* cos, const float* sin, const int* positions,
    const int* active, unsigned heads, unsigned dim, unsigned rotary, unsigned batch,
    unsigned context, unsigned normalize, float eps, float offset, unsigned slice, unsigned nblk,
    unsigned prefill = 0) {
    if (dim != 256 || (rotary != 0 && rotary != 64)) { if (threadIdx.x == 0) __trap(); return; }
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    for (unsigned row = slice * warps + warp; row < batch * heads; row += nblk * warps) {
        const unsigned slot = row / heads, head = row % heads;
        if (active && active[slot] <= 0) continue;
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
        size_t dest = context ? ((size_t)(prefill ? 0 : slot) * heads + head) * context * dim + (size_t)pos * dim
                              : (size_t)row * dim;
#pragma unroll
        for (unsigned j = 0; j < 8; j++) out[dest + lane + j * 32] = __float2bfloat16(x[j]);
    }
}

static __device__ void d_qwen_gdn_conv_prefill(__nv_bfloat16* out, const __nv_bfloat16* in,
    const __nv_bfloat16* weight, __nv_bfloat16* history, unsigned channels,
    unsigned width, unsigned tokens, unsigned slice, unsigned nblk) {
    if (width < 2 || !tokens) { if (threadIdx.x == 0) __trap(); return; }
    // One thread owns a channel and updates its history only after consuming the chunk.
    for (unsigned c = slice * blockDim.x + threadIdx.x; c < channels; c += nblk * blockDim.x) {
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
            out[(size_t)t * channels + c] = __float2bfloat16(sum / (1.f + expf(-sum)));
        }
        for (unsigned j = 0; j + 1 < width; j++) {
            int source = (int)tokens + (int)j - (int)(width - 1);
            h[j] = source < 0 ? h[tokens + j] : in[(size_t)source * channels + c];
        }
    }
}

static __device__ void d_qwen_gdn_qkv_prep(__nv_bfloat16* q, __nv_bfloat16* k,
    __nv_bfloat16* v, const __nv_bfloat16* packed, unsigned hk, unsigned hv,
    unsigned kd, unsigned vd, unsigned tokens, float eps, unsigned slice, unsigned nblk) {
    if (kd != 128 || !hk || !hv || !vd) { if (threadIdx.x == 0) __trap(); return; }
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    const unsigned channels = 2 * hk * kd + hv * vd;
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
