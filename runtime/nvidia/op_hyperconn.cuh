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
/* A plain (coherent) 16 B load as a volatile asm, so a batch of them issues before the first is
 * consumed; nvcc otherwise serializes them in the register-starved decode interpreter. */
__device__ __forceinline__ uint4 ld_u4(const uint16_t* p) {
    uint4 v;
    asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
__device__ __forceinline__ void cp16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((uint32_t)__cvta_generic_to_shared(dst)), "l"(src));
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

/* sinkhorn4 on a whole warp: lane & 15 owns c[i][j] (i = lane>>2 & 3, j = lane & 3), row sums
 * reduce over xor 1,2 and column sums over xor 4,8. One thread's serial loop costs ~28 us per call
 * in the decode interpreter; this is a few hundred cycles per iteration. Returns c[i][j]. */
__device__ __forceinline__ float sinkhorn4_warp(const float* __restrict__ mrow, float inv, const float* __restrict__ hc_scale,
                                                const float* __restrict__ hc_base, float hc_eps, unsigned repeat) {
    const unsigned e = threadIdx.x & 15u;
    auto sum_row = [](float v) {
        v += __shfl_xor_sync(0xffffffffu, v, 1);
        return v + __shfl_xor_sync(0xffffffffu, v, 2);
    };
    auto sum_col = [](float v) {
        v += __shfl_xor_sync(0xffffffffu, v, 4);
        return v + __shfl_xor_sync(0xffffffffu, v, 8);
    };
    float c = (mrow[8 + e] * inv) * hc_scale[2] + hc_base[8 + e];
    float m = fmaxf(c, __shfl_xor_sync(0xffffffffu, c, 1));
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
    c = expf(c - m);
    c = c / sum_row(c) + hc_eps;
    c = c / (sum_col(c) + hc_eps);
    for (unsigned r = 1; r < repeat; r++) {
        c = c / (sum_row(c) + hc_eps);
        c = c / (sum_col(c) + hc_eps);
    }
    return c;
}

}  // namespace plow_hc

/* t0=post_mix(out,[T,4]f32) t1=comb_mix(out,[T,4,4]f32) t2=layer_input(out,[T,hidden]bf16)
 * t3=mixes(in,[T,24]f32) t4=residual(in,[T,4,hidden]bf16) t5=hc_scale[3] t6=hc_base[24]
 * t7=pre_pair([2,T,4]f32) · i0=T i1=n i2=hidden i3=sinkhorn_repeat i4=pre_in_half i5=pre_mode
 * i6=mix_parts: > 1 means t3 holds GemvF32's K-slice partials [i6][T,24] (decode), summed here
 * j1=split (decode, one block per token): 0 = everything; bit 0 = ONLY t2, as the sublayer
 * RMSNorm of the collapse with gamma = tensor i7 (bit-identical to the RmsNorm op): the collapse
 * gates are the incoming pre (DEFER/SEED), so this reads neither t3 nor this sublayer's norm and
 * need not wait for the mixes GEMV; bit 1 = no post/comb (Sinkhorn); bit 2 = no collapse (pre,
 * post, comb only). The decode chain runs {bits 0|1} on its critical path and {bit 2} beside it. */
__device__ __forceinline__ void d_hyperconn_pre(float* __restrict__ post_mix, float* __restrict__ comb_mix, __nv_bfloat16* __restrict__ layer_input,
                                                const float* __restrict__ mixes, const __nv_bfloat16* __restrict__ residual,
                                                const float* __restrict__ hc_scale, const float* __restrict__ hc_base, unsigned T, unsigned n,
                                                unsigned hidden, unsigned repeat, float rms_eps, float hc_eps, unsigned slice, unsigned nblk,
                                                float* __restrict__ pre_pair, unsigned pre_in_half, unsigned pre_mode,
                                                unsigned mix_parts, float* arena = nullptr, unsigned arena_floats = 0,
                                                unsigned split = 0, const __nv_bfloat16* __restrict__ gamma = nullptr) {
    using namespace plow_hc;
    if (n != 4u || hidden % 8u) __trap();
    const unsigned warps = blockDim.x >> 5, wv = threadIdx.x >> 5, ln = threadIdx.x & 31u;
    const unsigned nh = 4u * hidden;
    const uint16_t* res = reinterpret_cast<const uint16_t*>(residual);
    if (T <= nblk && arena_floats >= nh / 2u + mix_parts * 24u) {
        /* Decode: a warp per token leaves the machine idle and walks 40 KB serially. `sub` BLOCKS
         * per token instead: each stages the row (and the GemvF32 partials) in the arena, takes the
         * norm, and collapses its slice of hidden; part 0 also publishes pre/post/comb, its last
         * warp running the Sinkhorn. The loops stay ROLLED with cp.async keeping every load in
         * flight: op bodies are cold in the decode interpreter's instruction cache each step. */
        /* s_par: mixes row [0,24), hc_base [24,48), hc_scale [48,51), incoming pre [51,55) */
        __shared__ float s_red[32], s_g[4], s_par[56], s_mp[10][24];
        const unsigned sub = nblk / T;
        if (slice >= T * sub) return;
        const bool mixless = split & 1u;
        if (split && (sub != 1u || (mixless && (!gamma || hidden / 8u > 3u * blockDim.x || pre_mode == PLOW_HC_PRE_OWN)))) __trap();
        const unsigned t = slice / sub, part = slice % sub, tid = threadIdx.x;
        const uint16_t* rrow = res + (size_t)t * nh;
        uint16_t* rs = reinterpret_cast<uint16_t*>(arena);
        float* ps = arena + nh / 2u;
        const unsigned rc = nh / 8u, pc = mix_parts > 1u && !mixless ? mix_parts * 6u : 0u;
#pragma unroll 1
        for (unsigned e = tid; e < rc + pc; e += blockDim.x) {
            if (e < rc)
                cp16(rs + e * 8u, rrow + e * 8u);
            else
                cp16(ps + (e - rc) * 4u, mixes + ((size_t)((e - rc) / 6u) * T + t) * 24u + ((e - rc) % 6u) * 4u);
        }
        asm volatile("cp.async.commit_group;\n" ::);
        const float pv = tid < 24u   ? (mix_parts > 1u ? 0.0f : mixes[(size_t)t * 24u + tid])
                         : tid < 48u ? hc_base[tid - 24u]
                         : tid < 51u ? hc_scale[tid - 48u]
                         : tid < 55u && pre_mode == PLOW_HC_PRE_DEFER ? pre_pair[(size_t)pre_in_half * T * 4u + (size_t)t * 4u + tid - 51u]
                                                                        : 0.0f;
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
        float ss = 0.0f;
#pragma unroll 1
        for (unsigned i = tid * 8u; i < (mixless ? 0u : nh); i += blockDim.x * 8u) {
            float v[8];
            unpack8(*reinterpret_cast<const uint4*>(rs + i), v);
#pragma unroll
            for (int u = 0; u < 8; u++) ss = fmaf(v[u], v[u], ss);
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        /* partials: thread (j, q) sums slices [q*per, (q+1)*per) of mix j in order, then j sums q */
        const unsigned mg = min(10u, blockDim.x / 24u), per = (mix_parts + mg - 1u) / mg;
        if (ln == 0) s_red[wv] = ss;
        if (tid < 56u) s_par[tid] = pv;
        if (mix_parts > 1u && !mixless && tid < mg * 24u) {
            const unsigned j = tid % 24u, q = tid / 24u;
            float mp = 0.0f;
#pragma unroll 1
            for (unsigned s0 = q * per; s0 < min(mix_parts, (q + 1u) * per); s0++) mp += ps[s0 * 24u + j];
            s_mp[q][j] = mp;
        }
        __syncthreads();
        if (mix_parts > 1u && !mixless) {
            if (tid < 24u) {
                float v = 0.0f;
                for (unsigned q = 0; q < mg; q++) v += s_mp[q][tid];
                s_par[tid] = v;
            }
            __syncthreads();
        }
        if (tid == 0 && mixless) {
#pragma unroll
            for (unsigned j = 0; j < 4u; j++) s_g[j] = pre_mode == PLOW_HC_PRE_SEED ? (j == 0 ? 1.0f : 0.0f) : s_par[51 + j];
        } else if (tid == 0) {
            float s = 0.0f;
            for (unsigned w2 = 0; w2 < warps; w2++) s += s_red[w2];
            s_red[31] = rsqrtf(s / (float)nh + rms_eps);
            const float inv = s_red[31];
            float g[4];
#pragma unroll
            for (unsigned j = 0; j < 4u; j++) g[j] = 1.0f / (1.0f + expf(-((s_par[j] * inv) * s_par[48] + s_par[24 + j]))) + hc_eps;
            if (pre_mode != PLOW_HC_PRE_OWN && part == 0) {
                float* po = pre_pair + (size_t)(pre_in_half ^ 1u) * T * 4u + (size_t)t * 4u;
#pragma unroll
                for (unsigned j = 0; j < 4u; j++) po[j] = g[j];
            }
#pragma unroll
            for (unsigned j = 0; j < 4u; j++)
                s_g[j] = pre_mode == PLOW_HC_PRE_OWN    ? g[j]
                         : pre_mode == PLOW_HC_PRE_SEED ? (j == 0 ? 1.0f : 0.0f)
                                                        : s_par[51 + j];
        }
        __syncthreads();
        if (part == 0 && wv == warps - 1u && !(split & 2u)) {
            const float inv = s_red[31];
            if (ln < 4u)
                post_mix[(size_t)t * 4u + ln] = (1.0f / (1.0f + expf(-((s_par[4 + ln] * inv) * s_par[49] + s_par[28 + ln])))) * PLOW_HC_POST_MULT;
            const float c = sinkhorn4_warp(s_par, inv, s_par + 48, s_par + 24, hc_eps, repeat);
            if (ln < 16u) comb_mix[(size_t)t * 16u + ln] = c;
        }
        const float g0 = s_g[0], g1 = s_g[1], g2 = s_g[2], g3 = s_g[3];
        uint16_t* lrow = reinterpret_cast<uint16_t*>(layer_input) + (size_t)t * hidden;
        const unsigned h8 = hidden / 8u;
        if (split & 4u) {
            __syncthreads(); /* the arena is the next op's */
            return;
        }
        if (split & 1u) {
            /* d_rmsnorm's block path on the bf16 collapse: the same thread -> chunk map
             * (tid + c * 256), sum order and formula, so the output is the RmsNorm op's bytes */
            uint4 xv[3];
            float ss2 = 0.0f;
#pragma unroll
            for (unsigned c = 0; c < 3u; c++) {
                const unsigned c8 = tid + c * blockDim.x;
                if (c8 < h8) {
                    float v0[8], v1[8], v2[8], v3[8], acc[8];
                    unpack8(*reinterpret_cast<const uint4*>(rs + c8 * 8u), v0);
                    unpack8(*reinterpret_cast<const uint4*>(rs + hidden + c8 * 8u), v1);
                    unpack8(*reinterpret_cast<const uint4*>(rs + 2u * hidden + c8 * 8u), v2);
                    unpack8(*reinterpret_cast<const uint4*>(rs + 3u * hidden + c8 * 8u), v3);
#pragma unroll
                    for (int u = 0; u < 8; u++) acc[u] = fmaf(g3, v3[u], fmaf(g2, v2[u], fmaf(g1, v1[u], g0 * v0[u])));
                    xv[c] = pack8(acc);
                    unpack8(xv[c], acc);
#pragma unroll
                    for (int u = 0; u < 8; u++) {
                        const float f = acc[u];
                        ss2 += f * f;
                    }
                }
            }
            const float inv2 = rsqrtf(block_sum(ss2, s_red) * __fdividef(1.0f, (float)hidden) + rms_eps);
            const uint16_t* gm = reinterpret_cast<const uint16_t*>(gamma);
#pragma unroll
            for (unsigned c = 0; c < 3u; c++) {
                const unsigned c8 = tid + c * blockDim.x;
                if (c8 < h8) {
                    float v[8], gw[8], o[8];
                    unpack8(xv[c], v);
                    unpack8(*reinterpret_cast<const uint4*>(gm + c8 * 8u), gw);
#pragma unroll
                    for (int u = 0; u < 8; u++) o[u] = __bfloat162float(__float2bfloat16(v[u] * inv2 * norm_weight(gw[u])));
                    *reinterpret_cast<uint4*>(lrow + c8 * 8u) = pack8(o);
                }
            }
            __syncthreads(); /* the arena is the next op's */
            return;
        }
#pragma unroll 1
        for (unsigned c8 = part * h8 / sub + tid; c8 < (part + 1u) * h8 / sub; c8 += blockDim.x) {
            float v0[8], v1[8], v2[8], v3[8], acc[8];
            unpack8(*reinterpret_cast<const uint4*>(rs + c8 * 8u), v0);
            unpack8(*reinterpret_cast<const uint4*>(rs + hidden + c8 * 8u), v1);
            unpack8(*reinterpret_cast<const uint4*>(rs + 2u * hidden + c8 * 8u), v2);
            unpack8(*reinterpret_cast<const uint4*>(rs + 3u * hidden + c8 * 8u), v3);
#pragma unroll
            for (int u = 0; u < 8; u++) acc[u] = fmaf(g3, v3[u], fmaf(g2, v2[u], fmaf(g1, v1[u], g0 * v0[u])));
            *reinterpret_cast<uint4*>(lrow + c8 * 8u) = pack8(acc);
        }
        __syncthreads(); /* the arena is the next op's */
        return;
    }
    if (mix_parts > 1u) __trap();
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
        uint4 rr[5];
#pragma unroll
        for (unsigned i = 0; i < 4u; i++) rr[i] = ld_u4(rs + ((size_t)t * 4u + i) * hidden + d);
        if (mode == 0u) rr[4] = ld_u4(xs + (size_t)t * hidden + d);
        float rv[4][8];
#pragma unroll
        for (unsigned i = 0; i < 4u; i++) unpack8(rr[i], rv[i]);
        if (mode == 2u) {
            float o[8];
#pragma unroll
            for (int u = 0; u < 8; u++) o[u] = (((rv[0][u] + rv[1][u]) + rv[2][u]) + rv[3][u]) / 4.0f;
            *reinterpret_cast<uint4*>(os + (size_t)t * hidden + d) = pack8(o);
            continue;
        }
        float xv[8];
        unpack8(rr[4], xv);
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
