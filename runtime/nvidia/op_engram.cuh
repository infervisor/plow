/* op_engram.cuh -- DeepSeek-V4.1 Engram on the NVIDIA interpreter: ops 209 (the gathered fp8 table
 * read) and 208 (gate + mix), term for term the AMD bodies in runtime/amd/op_engram.h, whose header
 * carries the reference derivation (model.py ParallelEngramEmbedding / Engram.forward).
 *
 * The table may live in device-mapped HOST memory (384M x 256 fp8 = 98 GB per table): a prefill
 * reads 24 x 256 B per token, so the gather crosses PCIe as ~6 KB per token.
 */
#pragma once
#include <cuda_bf16.h>
#include <stdint.h>

#define PLOW_ENGRAM_CLAMP 1e-6f

/* op 208: x [T][n][hidden] in place += gate * value, gate from the normalized dot of each hc copy
 * against its key. kv [T][(n+1)*hidden] = n keys then the shared value. tmask [T] optional. */
__device__ void d_engram_gate(__nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ kv, const __nv_bfloat16* __restrict__ qw,
                              const __nv_bfloat16* __restrict__ kw, const unsigned char* __restrict__ tmask, unsigned T, unsigned n,
                              unsigned hidden, float eps, unsigned slice, unsigned nblk, float* part) {
    const float inv_dim = 1.0f / (float)hidden, dim_rsqrt = rsqrtf((float)hidden);
    for (unsigned t = slice; t < T; t += nblk) {
        __nv_bfloat16* xrow = x + (size_t)t * n * hidden;
        const __nv_bfloat16* krow = kv + (size_t)t * (n + 1) * hidden;
        const __nv_bfloat16* vrow = krow + (size_t)n * hidden;
        const bool masked = tmask && tmask[t] == 0;
        for (unsigned c = 0; c < n; c++) {
            __nv_bfloat16* hc = xrow + (size_t)c * hidden;
            const __nv_bfloat16 *kc = krow + (size_t)c * hidden, *qc = qw + (size_t)c * hidden, *wc = kw + (size_t)c * hidden;
            float ssh = 0.f, ssk = 0.f, dot = 0.f;
            for (unsigned d = threadIdx.x; d < hidden; d += blockDim.x) {
                const float hv = __bfloat162float(hc[d]), kvv = __bfloat162float(kc[d]);
                ssh += hv * hv;
                ssk += kvv * kvv;
                dot += hv * (__bfloat162float(qc[d]) * __bfloat162float(wc[d])) * kvv;
            }
            ssh = block_sum(ssh, part);
            ssk = block_sum(ssk, part);
            dot = block_sum(dot, part);
            float gate = 0.f;
            if (!masked) {
                const float rstd = rsqrtf(ssh * inv_dim + eps) * rsqrtf(ssk * inv_dim + eps);
                const float dv = dot * rstd * dim_rsqrt;
                const float mag = sqrtf(fmaxf(fabsf(dv), PLOW_ENGRAM_CLAMP));
                gate = 1.0f / (1.0f + expf(-copysignf(mag, dv)));
            }
            if (gate != 0.f)
                for (unsigned d = threadIdx.x; d < hidden; d += blockDim.x)
                    hc[d] = __float2bfloat16(__bfloat162float(hc[d]) + gate * __bfloat162float(vrow[d]));
        }
    }
}

/* op 209: out [T][n_cols*head_dim] bf16 = dequant(table[ids - vocab_start]) with ue8m0 per blk;
 * ids outside this shard's [vocab_start, vocab_start + part_rows) write zeros. */
__device__ void d_engram_embed(__nv_bfloat16* __restrict__ out, const unsigned char* __restrict__ table,
                               const unsigned char* __restrict__ scale, const int* __restrict__ ids, unsigned T, unsigned n_cols,
                               unsigned head_dim, unsigned blk, unsigned vocab_start, unsigned part_rows, unsigned slice, unsigned nblk) {
    const unsigned width = n_cols * head_dim, spr = head_dim / blk;
    for (unsigned t = slice; t < T; t += nblk) {
        __nv_bfloat16* orow = out + (size_t)t * width;
        const int* irow = ids + (size_t)t * n_cols;
        /* 8 elements per thread: one 8 B table load, one scale byte (blk >= 8) */
        for (unsigned i = threadIdx.x * 8u; i < width; i += blockDim.x * 8u) {
            const unsigned col = i / head_dim, d = i % head_dim;
            const long long local = (long long)irow[col] - (long long)vocab_start;
            uint4 o = make_uint4(0, 0, 0, 0);
            if (local >= 0 && local < (long long)part_rows) {
                const size_t r = (size_t)local;
                const uint2 v = *reinterpret_cast<const uint2*>(table + r * head_dim + d);
                const float sc = __uint_as_float((uint32_t)scale[r * spr + d / blk] << 23);
                const uint32_t w[2] = {v.x, v.y};
                uint32_t p[4];
#pragma unroll
                for (int h = 0; h < 4; h++) {
                    const uint16_t pair = (uint16_t)(w[h >> 1] >> (16 * (h & 1)));
                    uint32_t h2;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"(pair));
                    float lo, hi;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(lo) : "h"((uint16_t)(h2 & 0xffffu)));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(hi) : "h"((uint16_t)(h2 >> 16)));
                    const __nv_bfloat162 b = __floats2bfloat162_rn(lo * sc, hi * sc);
                    p[h] = *reinterpret_cast<const uint32_t*>(&b);
                }
                o = make_uint4(p[0], p[1], p[2], p[3]);
            }
            *reinterpret_cast<uint4*>(orow + i) = o;
        }
    }
}

#if !PLOW_NV_SPEECH /* the speech object has its own op 173 (op_speech_f32.cuh) */
/* op 173: ids[r] = argmax_c x[r][c] over f32, lowest index on ties. One CTA per row. */
__device__ void d_argmax_f32(unsigned* __restrict__ ids, const float* __restrict__ x, unsigned rows, unsigned width, unsigned slice,
                             unsigned nblk, float* part) {
    for (unsigned r = slice; r < rows; r += nblk) {
        const float* xr = x + (size_t)r * width;
        float best = -INFINITY;
        unsigned bi = 0xffffffffu;
        for (unsigned c = threadIdx.x; c < width; c += blockDim.x) {
            const float v = xr[c];
            if (v > best || (v == best && c < bi)) best = v, bi = c;
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, best, o);
            const unsigned oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > best || (ov == best && oi < bi)) best = ov, bi = oi;
        }
        unsigned* pi = reinterpret_cast<unsigned*>(part + 32);
        if ((threadIdx.x & 31u) == 0) part[threadIdx.x >> 5] = best, pi[threadIdx.x >> 5] = bi;
        __syncthreads();
        if (threadIdx.x == 0) {
            for (unsigned w = 1; w < blockDim.x / 32u; w++)
                if (part[w] > best || (part[w] == best && pi[w] < bi)) best = part[w], bi = pi[w];
            ids[r] = bi;
        }
        __syncthreads();
    }
}
#endif
