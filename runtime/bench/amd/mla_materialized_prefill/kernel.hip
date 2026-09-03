#include <hip/hip_runtime.h>

#include "amd_common.h"
#include "op_attention.h"

static constexpr unsigned DK_ABS = 512;
static constexpr unsigned DR = 64;
static constexpr unsigned DK_MAT = 192;
static constexpr unsigned DV_MAT = 128;

template <int DK, int DV>
__device__ void d_materialized_lds_pipeline(
    bf16* __restrict__ out, const bf16* __restrict__ q, const bf16* __restrict__ k,
    const bf16* __restrict__ v, unsigned n_tok, unsigned n_head, unsigned kv_stride,
    float scale, bf16* lds) {
    constexpr unsigned BQ = 64;
    constexpr unsigned BKV = 128;
    constexpr unsigned QSTRIDE = DK + FA_PAD;
    constexpr unsigned KSTRIDE = DK + FA_PAD;
    constexpr unsigned VSTRIDE = DV + FA_PAD;
    constexpr unsigned QHALVES = BQ * QSTRIDE;
    constexpr unsigned KHALVES = BKV * KSTRIDE;
    constexpr unsigned VHALVES = BKV * VSTRIDE;
    constexpr unsigned NK = DK / 32;
    constexpr unsigned NDT = DV / 16;
    constexpr unsigned VPT = BKV * DV / PLOW_THREADS / 8;
    constexpr unsigned VOVERFLOW = VHALVES > QHALVES ? VHALVES - QHALVES : 0;
    static_assert(DK % MFMA_K == 0 && DV % MFMA_N == 0);
    static_assert(VOVERFLOW + BKV * BQ <= KHALVES,
                  "the consumed K slot must hold V overflow and P");

    bf16* qsm = lds;
    bf16* ksm[2] = {qsm + QHALVES, qsm + QHALVES + KHALVES};
    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 63;
    const unsigned wave = tid >> 6;
    const unsigned fr = lane & 15;
    const unsigned kg = lane >> 4;
    const unsigned qtiles = (n_tok + BQ - 1) / BQ;
    const unsigned nwork = qtiles * n_head;

    for (unsigned work = blockIdx.x; work < nwork; work += gridDim.x) {
        const unsigned h = work % n_head;
        const unsigned qt = mla_pf_fold(work / n_head, qtiles);
        const unsigned qbase = qt * BQ;
        const unsigned myq0 = qbase + wave * 16;

        for (unsigned e = tid * 8; e < BQ * DK; e += PLOW_THREADS * 8) {
            const unsigned r = e / DK, c = e % DK;
            bf16v8 x = bf16v8_zero();
            if (qbase + r < n_tok)
                x = ld_glob8(q + ((size_t)(qbase + r) * n_head + h) * DK + c);
            __builtin_memcpy(qsm + r * QSTRIDE + c, &x, 16);
        }
        for (unsigned e = tid * 8; e < BKV * DK; e += PLOW_THREADS * 8) {
            const unsigned r = e / DK, c = e % DK;
            bf16v8 x = bf16v8_zero();
            if (r < n_tok)
                x = ld_glob8(k + ((size_t)h * kv_stride + r) * DK + c);
            __builtin_memcpy(ksm[0] + r * KSTRIDE + c, &x, 16);
        }
        __syncthreads();

        f32x4 oacc[NDT];
#pragma unroll
        for (unsigned t = 0; t < NDT; ++t) oacc[t] = (f32x4)(0.0f);
        float mst[4], lst[4];
#pragma unroll
        for (unsigned i = 0; i < 4; ++i) { mst[i] = FA_NEG_INF; lst[i] = 0.0f; }

        unsigned slot = 0;
        for (unsigned kv0 = 0; kv0 < n_tok; kv0 += BKV) {
            f32x4 score[8];
#pragma unroll
            for (unsigned st = 0; st < 8; ++st) score[st] = (f32x4)(0.0f);
#pragma unroll
            for (unsigned kk = 0; kk < NK; ++kk) {
                bf16v8 nextk[2];
#pragma unroll
                for (unsigned it = 0; it < 2; ++it) {
                    const unsigned ne = tid * 8 + it * PLOW_THREADS * 8;
                    const unsigned nr = ne / 32, nc = ne % 32 + kk * 32;
                    nextk[it] = (kv0 + BKV + nr < n_tok)
                                    ? ld_glob8(k + ((size_t)h * kv_stride + kv0 + BKV + nr) * DK + nc)
                                    : bf16v8_zero();
                }
                bf16x8 qf;
                __builtin_memcpy(&qf, qsm + (wave * 16 + fr) * QSTRIDE + kk * 32 + kg * 8,
                                 16);
#pragma unroll
                for (unsigned st = 0; st < 8; ++st) {
                    bf16x8 kf;
                    __builtin_memcpy(&kf,
                                     ksm[slot] + (st * 16 + fr) * KSTRIDE + kk * 32 + kg * 8,
                                     16);
                    score[st] = plow_mfma_bf16_16x16(qf, kf, score[st]);
                }
#pragma unroll
                for (unsigned it = 0; it < 2; ++it) {
                    const unsigned ne = tid * 8 + it * PLOW_THREADS * 8;
                    const unsigned nr = ne / 32, nc = ne % 32 + kk * 32;
                    if (kv0 + BKV < n_tok)
                        __builtin_memcpy(ksm[slot ^ 1] + nr * KSTRIDE + nc, &nextk[it], 16);
                }
            }

            bf16v8 curv[VPT];
#pragma unroll
            for (unsigned it = 0; it < VPT; ++it) {
                const unsigned e = tid * 8 + it * PLOW_THREADS * 8;
                const unsigned r = e / DV, c = e % DV;
                curv[it] = (kv0 + r < n_tok)
                               ? ld_glob8(v + ((size_t)h * kv_stride + kv0 + r) * DV + c)
                               : bf16v8_zero();
            }

            float p[8][4];
#pragma unroll
            for (unsigned st = 0; st < 8; ++st) {
#pragma unroll
                for (unsigned i = 0; i < 4; ++i) {
                    const unsigned qi = myq0 + kg * 4 + i;
                    const unsigned krow = kv0 + st * 16 + fr;
                    const bool valid = qi < n_tok && krow < n_tok && krow <= qi;
                    p[st][i] = valid ? score[st][i] * FA_SCALE(scale) : FA_NEG_INF;
                }
            }
#pragma unroll
            for (unsigned i = 0; i < 4; ++i) {
                float rmax = FA_NEG_INF;
#pragma unroll
                for (unsigned st = 0; st < 8; ++st) {
                    float x = p[st][i];
#pragma unroll
                    for (unsigned d = 1; d < 16; d <<= 1) x = fmaxf(x, __shfl_xor(x, d, 64));
                    rmax = fmaxf(rmax, x);
                }
                const float mnew = fmaxf(mst[i], rmax);
                const float corr = mst[i] == FA_NEG_INF ? 0.0f : FA_EXP(mst[i] - mnew);
                float psum = 0.0f;
#pragma unroll
                for (unsigned st = 0; st < 8; ++st) {
                    const float pe = mnew == FA_NEG_INF ? 0.0f : FA_EXP(p[st][i] - mnew);
                    p[st][i] = pe;
                    float sm = pe;
#pragma unroll
                    for (unsigned d = 1; d < 16; d <<= 1) sm += __shfl_xor(sm, d, 64);
                    psum += sm;
                }
                lst[i] = lst[i] * corr + psum;
                mst[i] = mnew;
#pragma unroll
                for (unsigned t = 0; t < NDT; ++t) oacc[t][i] *= corr;
            }

            bf16* psm = ksm[slot] + VOVERFLOW;
            __syncthreads();
#pragma unroll
            for (unsigned st = 0; st < 8; ++st)
#pragma unroll
                for (unsigned i = 0; i < 4; ++i)
                    psm[(wave * 16 + kg * 4 + i) * BKV + st * 16 + fr] =
                        f2bf(p[st][i]);
#pragma unroll
            for (unsigned it = 0; it < VPT; ++it) {
                const unsigned e = tid * 8 + it * PLOW_THREADS * 8;
                const unsigned ve = (e / DV) * VSTRIDE + e % DV;
                bf16* dst = ve < QHALVES ? qsm + ve : ksm[slot] + ve - QHALVES;
                __builtin_memcpy(dst, &curv[it], 16);
            }
            __syncthreads();

            bf16x8 pf;
            __builtin_memcpy(&pf, psm + (wave * 16 + fr) * BKV + kg * 8, 16);
#pragma unroll
            for (unsigned t = 0; t < NDT; ++t) {
                bf16x8 vf;
#pragma unroll
                for (unsigned j = 0; j < 8; ++j) {
                        const unsigned ve = (kg * 8 + j) * VSTRIDE + t * 16 + fr;
                        const bf16 vv = ve < QHALVES ? qsm[ve] : ksm[slot][ve - QHALVES];
                        bf16_t vx;
                        __builtin_memcpy(&vx, &vv, 2);
                        vf[j] = vx;
                }
                oacc[t] = plow_mfma_bf16_16x16(pf, vf, oacc[t]);
            }

            __syncthreads();
            for (unsigned e = tid * 8; e < QHALVES; e += PLOW_THREADS * 8) {
                const unsigned r = e / QSTRIDE, c = e % QSTRIDE;
                bf16v8 x = bf16v8_zero();
                if (c + 8 <= DK && qbase + r < n_tok)
                    x = ld_glob8(q + ((size_t)(qbase + r) * n_head + h) * DK + c);
                __builtin_memcpy(qsm + e, &x, 16);
            }
            __syncthreads();
            slot ^= 1;
        }

#pragma unroll
        for (unsigned i = 0; i < 4; ++i) {
            const unsigned qi = myq0 + kg * 4 + i;
            if (qi >= n_tok) continue;
            const float inv = lst[i] > 0.0f ? FA_RECIP(lst[i]) : 0.0f;
#pragma unroll
            for (unsigned t = 0; t < NDT; ++t)
                out[((size_t)qi * n_head + h) * DV + t * 16 + fr] =
                    f2bf(oacc[t][i] * inv);
        }
        __syncthreads();
    }
}

extern "C" __global__ __launch_bounds__(256, 1) void k_absorbed(
    float* opart, float* mlpart, const bf16* qabs, const bf16* qrope, const bf16* ckv,
    const bf16* krope, const int* kv_len, unsigned n_tok, unsigned n_head,
    unsigned kv_stride, float scale) {
    __shared__ bf16 lds[FA_MLA_PF2_LDS_BYTES(DK_ABS, DR) / 2];
    d_flash_mla_prefill_v2<DK_ABS, DR>(
        opart, mlpart, qabs, qrope, ckv, krope, kv_len, 1, n_tok, n_head, kv_stride, 0,
        scale, PLOW_KV_MASK_NONE, blockIdx.x, gridDim.x, lds);
}

extern "C" __global__ __launch_bounds__(256, 1) void k_absorbed_fold(
    bf16* out, const float* opart, const float* mlpart, const bf16* wuv,
    unsigned n_tok, unsigned n_head) {
    __shared__ float lds[DK_ABS + PLOW_WAVES * PLOW_MLA_FOLD_VT];
    d_mla_merge_fold<DK_ABS, PLOW_MLA_FOLD_VT>(
        out, opart, mlpart, wuv, n_tok, n_head, DV_MAT, 1, blockIdx.x, gridDim.x, lds);
}

extern "C" __global__ __launch_bounds__(256, 1) void k_materialized(
    float* opart, float* mlpart, bf16* out, const bf16* q, const bf16* k,
    const bf16* v, unsigned n_tok, unsigned n_head, unsigned kv_stride, float scale) {
    __shared__ bf16 lds[FA_LDS_HALVES(DK_MAT)];
    d_flash_prefill<DK_MAT, false, DV_MAT>(
        opart, mlpart, out, q, k, v, n_tok, n_tok, n_head, n_head, 0, 0, scale,
        kv_stride, PLOW_KV_MASK_NONE, 1, blockIdx.x, gridDim.x, lds);
}

extern "C" __global__ __launch_bounds__(256, 1) void k_materialized_lds(
    bf16* out, const bf16* q, const bf16* k, const bf16* v, unsigned n_tok,
    unsigned n_head, unsigned kv_stride, float scale) {
    __shared__ bf16 lds[64 * (DK_MAT + FA_PAD) + 2 * 128 * (DK_MAT + FA_PAD)];
    d_materialized_lds_pipeline<DK_MAT, DV_MAT>(
        out, q, k, v, n_tok, n_head, kv_stride, scale, lds);
}

extern "C" __global__ void k_make_inputs(
    bf16* qabs, bf16* qrope, bf16* ckv, bf16* krope, bf16* qmat,
    bf16* kmat, bf16* vmat, bf16* wuv, unsigned n_tok, unsigned n_head) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    const size_t qrows = (size_t)n_tok * n_head;
    for (size_t e = i; e < qrows * DK_ABS; e += stride) {
        const unsigned d = e % DK_ABS;
        const unsigned x = (unsigned)e * 1664525u + 1013904223u;
        qabs[e] = d < DV_MAT ? (bf16)(0x3b00u | ((x >> 16) & 0xffu)) : (bf16)0;
    }
    for (size_t e = i; e < qrows * DR; e += stride) {
        const unsigned x = (unsigned)e * 22695477u + 1u;
        qrope[e] = (bf16)(0x3a80u | ((x >> 16) & 0x7fu));
    }
    for (size_t e = i; e < (size_t)n_tok * DK_ABS; e += stride) {
        const unsigned x = (unsigned)e * 1103515245u + 12345u;
        ckv[e] = (bf16)(0x3b00u | ((x >> 16) & 0xffu));
    }
    for (size_t e = i; e < (size_t)n_tok * DR; e += stride) {
        const unsigned x = (unsigned)e * 214013u + 2531011u;
        krope[e] = (bf16)(0x3a80u | ((x >> 16) & 0x7fu));
    }
    for (size_t e = i; e < qrows * DK_MAT; e += stride) {
        const size_t row = e / DK_MAT;
        const unsigned d = e % DK_MAT;
        qmat[e] = d < DV_MAT ? qabs[row * DK_ABS + d]
                             : qrope[row * DR + d - DV_MAT];
    }
    for (size_t e = i; e < (size_t)n_head * n_tok * DK_MAT; e += stride) {
        const size_t row = e / DK_MAT;
        const unsigned d = e % DK_MAT;
        const unsigned tok = row % n_tok;
        kmat[e] = d < DV_MAT ? ckv[(size_t)tok * DK_ABS + d]
                             : krope[(size_t)tok * DR + d - DV_MAT];
    }
    for (size_t e = i; e < (size_t)n_head * n_tok * DV_MAT; e += stride) {
        const unsigned d = e % DV_MAT;
        const unsigned tok = (e / DV_MAT) % n_tok;
        vmat[e] = ckv[(size_t)tok * DK_ABS + d];
    }
    for (size_t e = i; e < (size_t)n_head * DK_ABS * DV_MAT; e += stride) {
        const unsigned d = e % DV_MAT;
        const unsigned l = (e / DV_MAT) % DK_ABS;
        wuv[e] = (bf16)(l == d ? 0x3f80u : 0u);
    }
}

extern "C" __global__ void k_cache_flush(const unsigned* in, unsigned* out, size_t n) {
    unsigned v = 0;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        v ^= in[i];
    if (v) atomicXor(out, v);
}
