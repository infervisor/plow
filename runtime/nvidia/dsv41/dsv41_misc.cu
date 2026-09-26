// DeepSeek-V4.1 sm_90a: compressor, hyper-connections (mHC), Engram gate, embedding, sampling.
#include "dsv41_common.cuh"

// ---------------------------------------------------------------------------------------------------
// Compressor pooling (model.py Compressor.forward, ratio > 1). kv/score are fp32 [T][d].
// Prefill: group j pools rows j*r .. j*r+r-1 with a per-channel softmax over the r scores; the tail
// T % r rows go to the slot's state (rows 0..rem-1). out[j] = bf16(sum_r kv * softmax(score)).
// One thread per (group, channel).
DSV_EXTERN void dsv_compress_pool_prefill(bf16* __restrict__ out, const float* __restrict__ kv,
                                          const float* __restrict__ score, int T, int d, int ratio,
                                          float* __restrict__ st_kv, float* __restrict__ st_sc) {
    const int G = T / ratio, rem = T % ratio;
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (long long)G * d) {
        const int j = (int)(i / d), c = (int)(i % d);
        float mx = -INFINITY;
        for (int r = 0; r < ratio; r++) mx = fmaxf(mx, score[(long long)(j * ratio + r) * d + c]);
        float den = 0.f;
        for (int r = 0; r < ratio; r++) den += expf(score[(long long)(j * ratio + r) * d + c] - mx);
        float acc = 0.f;
        for (int r = 0; r < ratio; r++) {
            const long long e = (long long)(j * ratio + r) * d + c;
            acc += kv[e] * (expf(score[e] - mx) / den);
        }
        out[i] = f2bf(acc);
    } else if (i < (long long)G * d + (long long)rem * d) {
        const long long k = i - (long long)G * d;
        const int r = (int)(k / d), c = (int)(k % d);
        st_kv[(long long)r * d + c] = kv[(long long)(G * ratio + r) * d + c];
        st_sc[(long long)r * d + c] = score[(long long)(G * ratio + r) * d + c];
    }
}

// Compressor decode step for B slots: row b's token goes to state slot pos[b] % r; when that completes
// a group ((pos+1) % r == 0) the group is pooled into out[b]. st_* point at per-slot [r][d] state.
DSV_EXTERN void dsv_compress_pool_decode(bf16* __restrict__ out, const float* __restrict__ kv,
                                         const float* __restrict__ score, int B, int d, int ratio,
                                         const int* __restrict__ pos,
                                         const unsigned long long* __restrict__ st_kv_ptrs,
                                         const unsigned long long* __restrict__ st_sc_ptrs) {
    const int b = blockIdx.x;
    if (b >= B) return;
    float* skv = (float*)st_kv_ptrs[b];
    float* ssc = (float*)st_sc_ptrs[b];
    const int p = pos[b], slot = p % ratio;
    for (int c = threadIdx.x; c < d; c += blockDim.x) {
        skv[slot * d + c] = kv[(long long)b * d + c];
        ssc[slot * d + c] = score[(long long)b * d + c];
    }
    if ((p + 1) % ratio != 0) return;
    __syncthreads();
    for (int c = threadIdx.x; c < d; c += blockDim.x) {
        float mx = -INFINITY;
        for (int r = 0; r < ratio; r++) mx = fmaxf(mx, ssc[r * d + c]);
        float den = 0.f;
        for (int r = 0; r < ratio; r++) den += expf(ssc[r * d + c] - mx);
        float acc = 0.f;
        for (int r = 0; r < ratio; r++) acc += skv[r * d + c] * (expf(ssc[r * d + c] - mx) / den);
        out[(long long)b * d + c] = f2bf(acc);
    }
}

// Scatter rows into per-slot caches: dst_ptrs[b] + (row_base[b] + j) * row_bytes for row j of batch row b.
// rows[b] rows per batch row, packed contiguously in src in batch order.
DSV_EXTERN void dsv_scatter_rows(const uint8_t* __restrict__ src, int n_b, const int* __restrict__ rows,
                                 const int* __restrict__ src_row0, const unsigned long long* __restrict__ dst_ptrs,
                                 const int* __restrict__ row_base, int row_bytes) {
    const int b = blockIdx.y;
    if (b >= n_b) return;
    const int n = rows[b];
    uint8_t* dst = (uint8_t*)dst_ptrs[b];
    const long long total = (long long)n * row_bytes / 16;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
        const long long byte = i * 16;
        const int j = (int)(byte / row_bytes), off = (int)(byte % row_bytes);
        *(uint4*)(dst + ((long long)row_base[b] + j) * row_bytes + off) =
            *(const uint4*)(src + ((long long)src_row0[b] + j) * row_bytes + off);
    }
}

// Per batch row: copy rows[b] rows of row_bytes from src_ptrs[b] to dst_ptrs[b] (both already offset).
// Either side may be a peer device's memory (peer access enabled) -- the cross-stage cache mirrors.
DSV_EXTERN void dsv_copy_rows(const unsigned long long* __restrict__ dst_ptrs, const unsigned long long* __restrict__ src_ptrs,
                              const int* __restrict__ rows, int n_b, int row_bytes) {
    const int b = blockIdx.y;
    if (b >= n_b) return;
    const long long total = (long long)rows[b] * row_bytes / 16;
    uint4* d = (uint4*)dst_ptrs[b];
    const uint4* s = (const uint4*)src_ptrs[b];
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) d[i] = s[i];
}

// dst[i] = src[idx[i]] for rows of row_bytes (multiple of 16).
DSV_EXTERN void dsv_gather_rows(uint8_t* __restrict__ dst, const uint8_t* __restrict__ src, const int* __restrict__ idx, int n,
                                int row_bytes) {
    const int i = blockIdx.y;
    if (i >= n) return;
    const uint4* s = (const uint4*)(src + (long long)idx[i] * row_bytes);
    uint4* d = (uint4*)(dst + (long long)i * row_bytes);
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < row_bytes / 16; j += gridDim.x * blockDim.x) d[j] = s[j];
}

// ---------------------------------------------------------------------------------------------------
// mHC (Block.hc_mixes / hc_pre / hc_post, kernel.py hc_split_sinkhorn), hc = 4.
// rsq[t] = rsqrt(mean(x[t]^2) + eps) over the flattened hc*d stream (bf16 in, fp32 math).
DSV_EXTERN void dsv_row_rsqrt(float* __restrict__ rsq, const bf16* __restrict__ x, int n, float eps) {
    __shared__ float red[32];
    const bf16* xr = x + (long long)blockIdx.x * n;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = bf2f(xr[i]);
        ss += v * v;
    }
    ss = block_sum(ss, red);
    if (threadIdx.x == 0) rsq[blockIdx.x] = rsqrtf(ss / (float)n + eps);
}

// mixes[t][24] (raw F.linear) * rsq[t] -> pre[t][4], post[t][4], comb[t][4][4]. One thread per token.
DSV_EXTERN void dsv_hc_sinkhorn(float* __restrict__ pre, float* __restrict__ post, float* __restrict__ comb,
                                const float* __restrict__ mixes, const float* __restrict__ rsq,
                                const float* __restrict__ hc_scale, const float* __restrict__ hc_base, int T,
                                int iters, float eps) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    float m[24];
    const float r = rsq[t];
#pragma unroll
    for (int i = 0; i < 24; i++) m[i] = mixes[t * 24 + i] * r;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        pre[t * 4 + j] = 1.f / (1.f + expf(-(m[j] * hc_scale[0] + hc_base[j]))) + eps;
        post[t * 4 + j] = 2.f * (1.f / (1.f + expf(-(m[j + 4] * hc_scale[1] + hc_base[j + 4]))));
    }
    float c[4][4];
#pragma unroll
    for (int j = 0; j < 4; j++)
#pragma unroll
        for (int k = 0; k < 4; k++) c[j][k] = m[j * 4 + k + 8] * hc_scale[2] + hc_base[j * 4 + k + 8];
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const float mx = fmaxf(fmaxf(c[j][0], c[j][1]), fmaxf(c[j][2], c[j][3]));
        float s = 0.f;
#pragma unroll
        for (int k = 0; k < 4; k++) {
            c[j][k] = expf(c[j][k] - mx);
            s += c[j][k];
        }
#pragma unroll
        for (int k = 0; k < 4; k++) c[j][k] = c[j][k] / s + eps;
    }
    auto colnorm = [&]() {
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const float s = c[0][k] + c[1][k] + c[2][k] + c[3][k];
#pragma unroll
            for (int j = 0; j < 4; j++) c[j][k] = c[j][k] / (s + eps);
        }
    };
    colnorm();
    for (int it = 0; it < iters - 1; it++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const float s = c[j][0] + c[j][1] + c[j][2] + c[j][3];
#pragma unroll
            for (int k = 0; k < 4; k++) c[j][k] = c[j][k] / (s + eps);
        }
        colnorm();
    }
#pragma unroll
    for (int j = 0; j < 4; j++)
#pragma unroll
        for (int k = 0; k < 4; k++) comb[t * 16 + j * 4 + k] = c[j][k];
}

// y[t][d] = bf16(sum_c pre[t][c] * x[t][c][d])
DSV_EXTERN void dsv_hc_pre(bf16* __restrict__ y, const bf16* __restrict__ x, const float* __restrict__ pre, int T, int D) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)T * D) return;
    const int t = (int)(i / D), d = (int)(i % D);
    const bf16* xt = x + (long long)t * 4 * D + d;
    float acc = pre[t * 4 + 0] * bf2f(xt[0]);
    acc += pre[t * 4 + 1] * bf2f(xt[D]);
    acc += pre[t * 4 + 2] * bf2f(xt[2 * D]);
    acc += pre[t * 4 + 3] * bf2f(xt[3 * D]);
    y[i] = f2bf(acc);
}

// out[t][k][d] = bf16(post[t][k] * x[t][d] + sum_j comb[t][j][k] * res[t][j][d]). out may alias res only
// if every thread reads its four residual values before any write, which it does (one thread per (t,d)).
DSV_EXTERN void dsv_hc_post(bf16* __restrict__ out, const bf16* __restrict__ x, const bf16* __restrict__ res,
                            const float* __restrict__ post, const float* __restrict__ comb, int T, int D) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)T * D) return;
    const int t = (int)(i / D), d = (int)(i % D);
    const float xv = bf2f(x[i]);
    float r[4];
#pragma unroll
    for (int j = 0; j < 4; j++) r[j] = bf2f(res[((long long)t * 4 + j) * D + d]);
    const float* cb = comb + t * 16;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        float s = cb[0 * 4 + k] * r[0];
        s += cb[1 * 4 + k] * r[1];
        s += cb[2 * 4 + k] * r[2];
        s += cb[3 * 4 + k] * r[3];
        out[((long long)t * 4 + k) * D + d] = f2bf(post[t * 4 + k] * xv + s);
    }
}

// ---------------------------------------------------------------------------------------------------
// Engram gate + mix (model.py Engram.forward after wkv). kv[t] = [hc keys of D][shared value of D] bf16.
// x[t][c] <- bf16(h + gate * value), gate = sigmoid(signed sqrt(max(|dot|, 1e-6))), where
// dot = sum_d h*qw*kw*key * rsqrt(mean h^2 + eps) * rsqrt(mean key^2 + eps) * D^-0.5. One block per (t, c).
DSV_EXTERN void dsv_engram_gate(bf16* __restrict__ x, const bf16* __restrict__ kv, const bf16* __restrict__ qw,
                                const bf16* __restrict__ kw, const uint8_t* __restrict__ mask, int D, float eps) {
    __shared__ float red[32];
    const int t = blockIdx.x, c = blockIdx.y;
    bf16* h = x + ((long long)t * 4 + c) * D;
    const bf16* key = kv + (long long)t * 5 * D + (long long)c * D;
    const bf16* val = kv + (long long)t * 5 * D + 4LL * D;
    float hh = 0.f, kk = 0.f, dot = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        const float hv = bf2f(h[i]), kv_ = bf2f(key[i]);
        const float w = bf2f(qw[c * D + i]) * bf2f(kw[c * D + i]);
        hh += hv * hv;
        kk += kv_ * kv_;
        dot += hv * w * kv_;
    }
    hh = block_sum(hh, red);
    kk = block_sum(kk, red);
    dot = block_sum(dot, red);
    const float rstd = rsqrtf(hh / (float)D + eps) * rsqrtf(kk / (float)D + eps);
    const float dt = dot * rstd * rsqrtf((float)D);
    float gate = 1.f / (1.f + expf(-copysignf(sqrtf(fmaxf(fabsf(dt), 1e-6f)), dt)));
    if (mask && !mask[t]) gate = 0.f;
    __syncthreads();
    for (int i = threadIdx.x; i < D; i += blockDim.x) h[i] = f2bf(bf2f(h[i]) + gate * bf2f(val[i]));
}

// ---------------------------------------------------------------------------------------------------
// x[t][c][:] = embed[ids[t]] for all 4 hc copies.
DSV_EXTERN void dsv_embed_hc(bf16* __restrict__ x, const bf16* __restrict__ emb, const int* __restrict__ ids, int T, int D) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)T * D / 8) return;
    const int t = (int)(i / (D / 8)), c8 = (int)(i % (D / 8)) * 8;
    const uint4 v = *(const uint4*)(emb + (long long)ids[t] * D + c8);
#pragma unroll
    for (int c = 0; c < 4; c++) *(uint4*)(x + ((long long)t * 4 + c) * D + c8) = v;
}

// out[b] = argmax_j logits[b][j] (first index on ties), one block per row.
DSV_EXTERN void dsv_argmax(int* __restrict__ out, const float* __restrict__ logits, int V) {
    __shared__ float sv[32];
    __shared__ int si[32];
    const float* l = logits + (long long)blockIdx.x * V;
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    for (int j = threadIdx.x; j < V; j += blockDim.x) {
        const float v = l[j];
        if (v > bv) {
            bv = v;
            bi = j;
        }
    }
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) {
            bv = ov;
            bi = oi;
        }
    }
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (lane == 0) {
        sv[w] = bv;
        si[w] = bi;
    }
    __syncthreads();
    if (w == 0) {
        const int nw = blockDim.x >> 5;
        bv = lane < nw ? sv[lane] : -INFINITY;
        bi = lane < nw ? si[lane] : 0x7fffffff;
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) {
                bv = ov;
                bi = oi;
            }
        }
        if (lane == 0) out[blockIdx.x] = bi;
    }
}

// Elementwise helpers.
DSV_EXTERN void dsv_bf16_to_f32(float* __restrict__ y, const bf16* __restrict__ x, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = bf2f(x[i]);
}
DSV_EXTERN void dsv_scale_bf16(bf16* __restrict__ x, long long n, float s) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = f2bf(bf2f(x[i]) * s);
}
