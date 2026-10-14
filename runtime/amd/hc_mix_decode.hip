/* Standalone one-row decode mHC mix (PLOW_GLM_HC_MIX_RAW; GemvF32 i3 = 1, i4 = 3, i5 = 0,
 * i6 bit 0, i7 = PLOW_HC_MIX_RAW_MARK): the interpreter's exec_hc_fuse_dec R = 0 arm with the
 * lean tail. Workgroup n forms output n (workgroup 0 also stores the new residual); the last
 * arrival runs the pre + norm tail. HC_MIX_UNROLL (default) issues every step's loads at once
 * and feeds the tail the residual from registers; HC_MIX_UNROLL=0 calls the interpreter bodies
 * (d_gemv_f32_wg_hcpost, hc_pre_norm_row_lean). Both are bit-identical to the packet. */
#include "amd_common.h"
#include "op_hyperconn.h"

extern "C" __device__ unsigned plow_hc_mix_decode_1 = 1;

#ifndef HC_MIX_UNROLL
#define HC_MIX_UNROLL 1
#endif

#if HC_MIX_UNROLL
/* hc_pre_norm_row_lean (op_hyperconn.h) with the row's four streams passed in registers: the
 * unrolled mix leaves thread t holding stream i at channels 8t..8t+7 as its step q = i value,
 * exactly the block the lean tail loads. Same expressions and order. */
__device__ __forceinline__ void hc_mix_tail_rv(const bf16v8 (&rv)[4], float* __restrict__ post_mix, float* __restrict__ comb_mix,
                                     bf16* __restrict__ lin, bf16* __restrict__ nout,
                                     const bf16* __restrict__ gamma,
                                     const float* __restrict__ mrow,
                                     const float* __restrict__ hc_scale,
                                     const float* __restrict__ hc_base, unsigned hidden,
                                     unsigned sinkhorn_repeat, float rms_eps, float hc_eps,
                                     float* part) {
    const unsigned nh = 4u * hidden;
    const unsigned wave = threadIdx.x / PLOW_WAVE, lane = threadIdx.x % PLOW_WAVE;
    const unsigned d = threadIdx.x * 8u;
    __shared__ float lw[PLOW_WAVES];
    __shared__ float lg[4];
    const bf16v8 gw = nout ? ld_glob8(gamma + d) : bf16v8_zero();
    float ss = 0.0f;
#pragma unroll
    for (unsigned i = 0; i < 4; i++)
#pragma unroll
        for (unsigned e = 0; e < 8; e++) {
            const float x = bf2f(rv[i][e]);
            ss += x * x;
        }
    ss = wave_sum(ss);
    if (lane == 0) lw[wave] = ss;
    __syncthreads();
    if (threadIdx.x < 16u) {
        const unsigned e = threadIdx.x;
        const unsigned rb = (lane & ~15u) | (e & ~3u), cb = (lane & ~15u) | (e & 3u);
        float s = 0.0f;
#pragma unroll
        for (int i = 0; i < PLOW_WAVES; i++) s += lw[i];
        const float inv = rsqrtf(s / (float)nh + rms_eps);
        if (e < 4u) {
            const float v = (mrow[e] * inv) * hc_scale[0] + hc_base[e];
            lg[e] = 1.0f / (1.0f + expf(-v)) + hc_eps;
            const float w = (mrow[4 + e] * inv) * hc_scale[1] + hc_base[4 + e];
            post_mix[e] = (1.0f / (1.0f + expf(-w))) * PLOW_HC_POST_MULT;
        }
        float c = (mrow[8 + e] * inv) * hc_scale[2] + hc_base[8 + e];
        float m = -3.0e38f;
#pragma unroll
        for (unsigned q = 0; q < 4; q++) m = fmaxf(m, __shfl(c, rb + q, PLOW_WAVE));
        c = expf(c - m);
        s = 0.0f;
#pragma unroll
        for (unsigned q = 0; q < 4; q++) s += __shfl(c, rb + q, PLOW_WAVE);
        c = c / s + hc_eps;
        s = 0.0f;
#pragma unroll
        for (unsigned q = 0; q < 4; q++) s += __shfl(c, cb + 4u * q, PLOW_WAVE);
        c = c / (s + hc_eps);
        for (unsigned r = 1; r < sinkhorn_repeat; r++) {
            s = 0.0f;
#pragma unroll
            for (unsigned q = 0; q < 4; q++) s += __shfl(c, rb + q, PLOW_WAVE);
            c = c / (s + hc_eps);
            s = 0.0f;
#pragma unroll
            for (unsigned q = 0; q < 4; q++) s += __shfl(c, cb + 4u * q, PLOW_WAVE);
            c = c / (s + hc_eps);
        }
        comb_mix[e] = c;
    }
    __syncthreads();
    const float l0 = lg[0], l1 = lg[1], l2 = lg[2], l3 = lg[3];
    bf16v8 o;
#pragma unroll
    for (unsigned e = 0; e < 8; e++) {
        float acc = 0.0f;
        acc += l0 * bf2f(rv[0][e]);
        acc += l1 * bf2f(rv[1][e]);
        acc += l2 * bf2f(rv[2][e]);
        acc += l3 * bf2f(rv[3][e]);
        o[e] = f2bf(acc);
    }
    st_glob8(lin + d, o);
    if (nout) {
        float s2 = 0.0f;
#pragma unroll
        for (unsigned j = 0; j < 8; j++) {
            const float f = bf2f(o[j]);
            s2 += f * f;
        }
        const float inv = rsqrtf(rn_ss(block_sum(s2, part)) / (float)hidden + rms_eps);
        bf16v8 q;
#pragma unroll
        for (unsigned j = 0; j < 8; j++) q[j] = f2bf(bf2f(o[j]) * inv * bf2f(gw[j]));
        st_glob8(nout + d, q);
    }
    __syncthreads(); /* lw / lg reused by the next row */
}
#endif

extern "C" __global__ __launch_bounds__(PLOW_THREADS)
void plow_hc_mix_decode_r1(float* mixes, bf16* new_residual, const float* W, const bf16* x_out,
                           const bf16* residual, float* post_mix, float* comb_mix, bf16* lin,
                           bf16* nout, const bf16* gamma, const float* hc_scale,
                           const float* hc_base, unsigned* ctr, unsigned K, unsigned hidden,
                           unsigned sink, float eps, float hceps, unsigned norm) {
    constexpr unsigned N = 24u;
    if (blockDim.x != PLOW_THREADS || gridDim.x != N || hidden != 8u * PLOW_THREADS ||
        K != 4u * hidden || PLOW_WAVES * 3u != N)
        __builtin_trap();
    __shared__ float part[4u * PLOW_THREADS];
    __shared__ float red[PLOW_WAVES];
    __shared__ unsigned last;
#if HC_MIX_UNROLL
    bf16v8 xs[4];
    {
        /* d_gemv_f32_wg_hcpost at M = 1, K = 16384 with the per-wave steps fully unrolled so
         * every load is in flight at once. Same terms, same step-order adds. */
        constexpr unsigned KK = 4u * 8u * PLOW_THREADS, steps = KK / (PLOW_WAVE * 8u);
        constexpr unsigned hid = KK / 4u;
        const unsigned wave = threadIdx.x / PLOW_WAVE, lane = threadIdx.x % PLOW_WAVE;
        const unsigned n = blockIdx.x;
        const float* const wn = W + (size_t)n * KK;
        float c[16], p[4];
#pragma unroll
        for (unsigned k = 0; k < 16; k++) c[k] = comb_mix[k];
#pragma unroll
        for (unsigned k = 0; k < 4; k++) p[k] = post_mix[k];
        constexpr unsigned per = steps / PLOW_WAVES;
        static_assert(per == 4u, "step q of thread t is stream q at channels 8t..8t+7");
        bf16v8 r[per][4], xo[per];
        float4 w0[per], w1[per];
#pragma unroll
        for (unsigned q = 0; q < per; q++) {
            const unsigned k = (wave + q * PLOW_WAVES) * (PLOW_WAVE * 8u) + lane * 8u;
            const unsigned d = k % hid;
#pragma unroll
            for (unsigned i = 0; i < 4; i++) r[q][i] = ld_glob8(residual + (size_t)i * hid + d);
            xo[q] = ld_glob8(x_out + d);
            w0[q] = *(const float4*)(wn + k);
            w1[q] = *(const float4*)(wn + k + 4);
        }
#pragma unroll
        for (unsigned q = 0; q < per; q++) {
            const unsigned s = wave + q * PLOW_WAVES;
            const unsigned k = s * (PLOW_WAVE * 8u) + lane * 8u;
            const bf16v8 xv = hc_post4_8(c, p, k / hid, r[q], xo[q]);
            xs[q] = xv;
            if (n == 0u) st_glob8(new_residual + k, xv);
            {
#pragma clang fp contract(off)
                part[s * PLOW_WAVE + lane] =
                    bf2f(xv[0]) * w0[q].x + bf2f(xv[1]) * w0[q].y + bf2f(xv[2]) * w0[q].z +
                    bf2f(xv[3]) * w0[q].w + bf2f(xv[4]) * w1[q].x + bf2f(xv[5]) * w1[q].y +
                    bf2f(xv[6]) * w1[q].z + bf2f(xv[7]) * w1[q].w;
            }
        }
        __syncthreads();
        if (wave == 0) {
#pragma clang fp contract(off)
            float acc = 0.0f;
            for (unsigned s = 0; s < steps; s++) acc += part[s * PLOW_WAVE + lane];
            acc = wave_sum(acc);
            if (lane == 0) mixes[n] = acc;
        }
    }
#else
    d_gemv_f32_wg_hcpost(mixes, new_residual, W, x_out, residual, post_mix, comb_mix, 1u, N, K,
                         blockIdx.x, N, part);
#endif
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned old =
            __hip_atomic_fetch_add(ctr, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        last = old == N - 1u;
        if (last) __hip_atomic_store(ctr, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    __syncthreads();
    if (!last) return;
    asm volatile("buffer_inv" ::: "memory");
    __syncthreads();
#if HC_MIX_UNROLL
    hc_mix_tail_rv(xs, post_mix, comb_mix, lin, norm ? nout : nullptr, norm ? gamma : nullptr,
                   mixes, hc_scale, hc_base, hidden, sink, eps, hceps, red);
#else
    hc_pre_norm_row_lean(post_mix, comb_mix, lin, norm ? nout : nullptr, norm ? gamma : nullptr,
                         mixes, new_residual, hc_scale, hc_base, hidden, sink, eps, hceps, red);
#endif
}
