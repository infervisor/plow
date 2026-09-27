/* op_hyperconn.cuh -- hyper-connections (mHC) on the warp32 interpreters: PLOW_DOP_HYPER_CONN_PRE
 * (128) and PLOW_DOP_HYPER_CONN_POST (129). Same operands and semantics as runtime/amd/op_hyperconn.h
 * (read its notes on pre_mode and the Sinkhorn); n = 4 only, which is every model that emits them.
 *
 * pre: ONE WARP PER TOKEN (the AMD file's measured fix for the serial Sinkhorn): the sum of squares
 * over n*hidden is a warp reduction, lane 0 runs the register-resident 4x4 Sinkhorn, the collapse
 * gate reaches the other lanes by shuffle, and the collapse streams 8 columns per lane.
 * post: one thread per 8 columns of a token, the four residual rows read once each.
 */
#pragma once
#include <cuda_bf16.h>
#include <stdint.h>

#define PLOW_HC_POST_MULT 2.0f
#define PLOW_HC_PRE_OWN 0u
#define PLOW_HC_PRE_SEED 1u
#define PLOW_HC_PRE_DEFER 2u

namespace plow_hc {

__device__ __forceinline__ float bf(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }
__device__ __forceinline__ uint16_t to_bf(float f) {
    const __nv_bfloat16 v = __float2bfloat16_rn(f);
    return *(const uint16_t*)&v;
}
__device__ __forceinline__ void unpack8(uint4 v, float* o) {
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int q = 0; q < 4; q++) {
        o[2 * q] = __uint_as_float(w[q] << 16);
        o[2 * q + 1] = __uint_as_float(w[q] & 0xffff0000u);
    }
}
__device__ __forceinline__ uint4 pack8(const float* o) {
    uint32_t w[4];
#pragma unroll
    for (int q = 0; q < 4; q++) w[q] = (uint32_t)to_bf(o[2 * q]) | ((uint32_t)to_bf(o[2 * q + 1]) << 16);
    return make_uint4(w[0], w[1], w[2], w[3]);
}

/* softmax over each row (+eps), one column normalize, then repeat-1 (row, column) pairs --
 * mhc_pre_torch's loop, as runtime/amd/op_hyperconn.h hc_sinkhorn4 */
__device__ __forceinline__ void sinkhorn4(float* c, const float* __restrict__ mrow, float inv, const float* __restrict__ hc_scale,
                                          const float* __restrict__ hc_base, float hc_eps, unsigned repeat) {
#pragma unroll
    for (unsigned i = 0; i < 4u; i++) {
        float m = -3.0e38f;
#pragma unroll
        for (unsigned j = 0; j < 4u; j++) {
            const float v = (mrow[8 + i * 4 + j] * inv) * hc_scale[2] + hc_base[8 + i * 4 + j];
            c[i * 4 + j] = v;
            m = fmaxf(m, v);
        }
        float s = 0.0f;
#pragma unroll
        for (unsigned j = 0; j < 4u; j++) {
            const float e = expf(c[i * 4 + j] - m);
            c[i * 4 + j] = e;
            s += e;
        }
#pragma unroll
        for (unsigned j = 0; j < 4u; j++) c[i * 4 + j] = c[i * 4 + j] / s + hc_eps;
    }
    auto cols = [&]() {
#pragma unroll
        for (unsigned j = 0; j < 4u; j++) {
            float s = 0.0f;
#pragma unroll
            for (unsigned i = 0; i < 4u; i++) s += c[i * 4 + j];
#pragma unroll
            for (unsigned i = 0; i < 4u; i++) c[i * 4 + j] = c[i * 4 + j] / (s + hc_eps);
        }
    };
    cols();
    for (unsigned r = 1; r < repeat; r++) {
#pragma unroll
        for (unsigned i = 0; i < 4u; i++) {
            float s = 0.0f;
#pragma unroll
            for (unsigned j = 0; j < 4u; j++) s += c[i * 4 + j];
#pragma unroll
            for (unsigned j = 0; j < 4u; j++) c[i * 4 + j] = c[i * 4 + j] / (s + hc_eps);
        }
        cols();
    }
}

}  // namespace plow_hc

/* t0=post_mix(out,[T,4]f32) t1=comb_mix(out,[T,4,4]f32) t2=layer_input(out,[T,hidden]bf16)
 * t3=mixes(in,[T,24]f32) t4=residual(in,[T,4,hidden]bf16) t5=hc_scale[3] t6=hc_base[24]
 * t7=pre_pair([2,T,4]f32) · i0=T i1=n i2=hidden i3=sinkhorn_repeat i4=pre_in_half i5=pre_mode */
__device__ __forceinline__ void d_hyperconn_pre(float* __restrict__ post_mix, float* __restrict__ comb_mix, __nv_bfloat16* __restrict__ layer_input,
                                                const float* __restrict__ mixes, const __nv_bfloat16* __restrict__ residual,
                                                const float* __restrict__ hc_scale, const float* __restrict__ hc_base, unsigned T, unsigned n,
                                                unsigned hidden, unsigned repeat, float rms_eps, float hc_eps, unsigned slice, unsigned nblk,
                                                float* __restrict__ pre_pair, unsigned pre_in_half, unsigned pre_mode) {
    using namespace plow_hc;
    if (n != 4u || hidden % 8u) __trap();
    const unsigned warps = blockDim.x >> 5, wv = threadIdx.x >> 5, ln = threadIdx.x & 31u;
    const unsigned nh = 4u * hidden;
    const uint16_t* res = reinterpret_cast<const uint16_t*>(residual);
    for (unsigned t = slice * warps + wv; t < T; t += nblk * warps) {
        const float* mrow = mixes + (size_t)t * 24u;
        const uint16_t* rrow = res + (size_t)t * nh;
        float ss = 0.0f;
        for (unsigned i = ln * 8u; i < nh; i += 256u) {
            float v[8];
            unpack8(*reinterpret_cast<const uint4*>(rrow + i), v);
#pragma unroll
            for (int u = 0; u < 8; u++) ss = fmaf(v[u], v[u], ss);
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        const float inv = rsqrtf(ss / (float)nh + rms_eps);
        float g[4] = {0.f, 0.f, 0.f, 0.f};
        if (ln == 0) {
#pragma unroll
            for (unsigned j = 0; j < 4u; j++) g[j] = 1.0f / (1.0f + expf(-((mrow[j] * inv) * hc_scale[0] + hc_base[j]))) + hc_eps;
            if (pre_mode != PLOW_HC_PRE_OWN) {
                float* po = pre_pair + (size_t)(pre_in_half ^ 1u) * T * 4u + (size_t)t * 4u;
#pragma unroll
                for (unsigned j = 0; j < 4u; j++) po[j] = g[j];
            }
#pragma unroll
            for (unsigned j = 0; j < 4u; j++)
                post_mix[(size_t)t * 4u + j] = (1.0f / (1.0f + expf(-((mrow[4 + j] * inv) * hc_scale[1] + hc_base[4 + j])))) * PLOW_HC_POST_MULT;
            float c[16];
            sinkhorn4(c, mrow, inv, hc_scale, hc_base, hc_eps, repeat);
#pragma unroll
            for (unsigned k = 0; k < 16u; k++) comb_mix[(size_t)t * 16u + k] = c[k];
        }
#pragma unroll
        for (unsigned j = 0; j < 4u; j++)
            g[j] = pre_mode == PLOW_HC_PRE_OWN    ? __shfl_sync(0xffffffffu, g[j], 0)
                   : pre_mode == PLOW_HC_PRE_SEED ? (j == 0 ? 1.0f : 0.0f)
                                                  : pre_pair[(size_t)pre_in_half * T * 4u + (size_t)t * 4u + j];
        uint16_t* lrow = reinterpret_cast<uint16_t*>(layer_input) + (size_t)t * hidden;
        for (unsigned d = ln * 8u; d < hidden; d += 256u) {
            float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (unsigned i = 0; i < 4u; i++) {
                float v[8];
                unpack8(*reinterpret_cast<const uint4*>(rrow + (size_t)i * hidden + d), v);
#pragma unroll
                for (int u = 0; u < 8; u++) acc[u] = fmaf(g[i], v[u], acc[u]);
            }
            *reinterpret_cast<uint4*>(lrow + d) = pack8(acc);
        }
    }
}

/* t0=new_residual(out,[T,4,hidden]) t1=x_out(in,[T,hidden]) t2=residual(in,[T,4,hidden])
 * t3=post_mix([T,4]f32) t4=comb_mix([T,4,4]f32) · i0=T i1=n i2=hidden i3=mode
 * mode 0: new[j] = sum_i comb[i][j] * res[i] + post[j] * x; 1: replicate x; 2: mean of the streams */
__device__ __forceinline__ void d_hyperconn_post(__nv_bfloat16* __restrict__ new_residual, const __nv_bfloat16* __restrict__ x_out,
                                                 const __nv_bfloat16* __restrict__ residual, const float* __restrict__ post_mix,
                                                 const float* __restrict__ comb_mix, unsigned T, unsigned n, unsigned hidden, unsigned mode,
                                                 unsigned slice, unsigned nblk) {
    using namespace plow_hc;
    if (n != 4u || hidden % 8u) __trap();
    const unsigned h8 = hidden / 8u;
    const uint16_t* xs = reinterpret_cast<const uint16_t*>(x_out);
    const uint16_t* rs = reinterpret_cast<const uint16_t*>(residual);
    uint16_t* os = reinterpret_cast<uint16_t*>(new_residual);
    const size_t total = (size_t)T * h8;
    for (size_t c = (size_t)slice * blockDim.x + threadIdx.x; c < total; c += (size_t)nblk * blockDim.x) {
        const unsigned t = (unsigned)(c / h8), d = (unsigned)(c % h8) * 8u;
        if (mode == 1u) {
            const uint4 xv = *reinterpret_cast<const uint4*>(xs + (size_t)t * hidden + d);
#pragma unroll
            for (unsigned j = 0; j < 4u; j++) *reinterpret_cast<uint4*>(os + ((size_t)t * 4u + j) * hidden + d) = xv;
            continue;
        }
        float rv[4][8];
#pragma unroll
        for (unsigned i = 0; i < 4u; i++) unpack8(*reinterpret_cast<const uint4*>(rs + ((size_t)t * 4u + i) * hidden + d), rv[i]);
        if (mode == 2u) {
            float o[8];
#pragma unroll
            for (int u = 0; u < 8; u++) o[u] = (((rv[0][u] + rv[1][u]) + rv[2][u]) + rv[3][u]) / 4.0f;
            *reinterpret_cast<uint4*>(os + (size_t)t * hidden + d) = pack8(o);
            continue;
        }
        float xv[8];
        unpack8(*reinterpret_cast<const uint4*>(xs + (size_t)t * hidden + d), xv);
        const float* pm = post_mix + (size_t)t * 4u;
        const float* cm = comb_mix + (size_t)t * 16u;
#pragma unroll
        for (unsigned j = 0; j < 4u; j++) {
            float o[8];
#pragma unroll
            for (int u = 0; u < 8; u++) {
                float acc = 0.0f;
#pragma unroll
                for (unsigned i = 0; i < 4u; i++) acc += cm[i * 4 + j] * rv[i][u];
                o[u] = acc + pm[j] * xv[u];
            }
            *reinterpret_cast<uint4*>(os + ((size_t)t * 4u + j) * hidden + d) = pack8(o);
        }
    }
}
