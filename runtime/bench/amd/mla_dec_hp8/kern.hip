// GLM-5.3-Flash TP8 batched decode MLA (NoPE, 8 heads, latent 512, V 256) bodies of the
// decode-MLA object as standalone kernels: the hp8 flash, the scalar fold and the MFMA fold.
// -DHP8_BKV=<64|128>: the hp8 flash with a wider KV tile (hp8_wide below).
#include "amd_common.h"
#include "op_attention.h"
#include "mla_fold_pf_mfma.h"

#ifndef HP8_BKV
#define HP8_BKV 32
#endif
#ifndef HP8_PD
#define HP8_PD 1
#endif

/* d_flash_mla_decode_hp8 with BKV keys per tile: NB = BKV/32 key blocks per QK k-step, the PV
 * contraction over BKV keys. Same partial layout. */
template <int DK, int BKV, int PD>
__device__ void hp8_wide(float* __restrict__ Opart, float* __restrict__ mlpart,
                         const bf16* __restrict__ Qabs, const bf16* __restrict__ Ckv,
                         const int* __restrict__ kv_len, unsigned n_batch, unsigned kv_stride,
                         float scale, unsigned nsplit, unsigned slice, unsigned nblk, float* lds_) {
    constexpr int NH = 8;
    constexpr int NB = BKV / 32;
    constexpr int STRIDE = DK + FA_PAD;
    constexpr int NK = DK / MFMA_K;
    constexpr int KPW = NK / PLOW_WAVES;
    constexpr int CPW = DK / PLOW_WAVES;
    constexpr int NDT = CPW / MFMA_N;
    constexpr int CH = DK / 8;
    constexpr int TPT = BKV * CH / PLOW_THREADS;

    bf16* Ksm = (bf16*)lds_;
    bf16* Qsm = Ksm + BKV * STRIDE;
    bf16* Psm = Qsm + NH * STRIDE;                       /* [NH][BKV] */
    float* Sred = (float*)(Psm + NH * BKV);              /* [WAVES][NH][BKV] */

    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 63;
    const unsigned wave = tid >> 6;
    const unsigned frow = mfma_frag_row(lane);
    const unsigned accn = mfma_acc_n(lane);
    const unsigned ncol0 = wave * CPW;
    const bf16x8 zero8 = (bf16x8)(bf16_t)0.0f;

    const unsigned n_work = n_batch * nsplit;
    for (unsigned w = slice; w < n_work; w += nblk) {
        const unsigned sp = w % nsplit;
        const unsigned b = w / nsplit;
        const unsigned len = (unsigned)kv_len[b];
        const unsigned per = (len + nsplit - 1) / nsplit;
        const unsigned lo = sp * per < len ? sp * per : len;
        const unsigned hi = lo + per < len ? lo + per : len;
        const auto* cbase = as_glob(Ckv) + (size_t)b * kv_stride * DK;

        __syncthreads();
        if (tid < NH * CH) {
            const unsigned h = tid / CH, c = tid % CH;
            *(bf16v8*)&Qsm[h * STRIDE + c * 8] = ld_glob8(as_glob(Qabs) + ((size_t)b * NH + h) * DK + c * 8);
        }
        bf16v8 nxt[PD][TPT];
        auto fetch = [&](bf16v8 (&dst)[TPT], unsigned kv0) {
#pragma unroll
            for (int j = 0; j < TPT; j++) {
                const unsigned e = tid + (unsigned)j * PLOW_THREADS;
                const unsigned kv = kv0 + e / CH;
                dst[j] = kv < hi ? ld_glob8(cbase + (size_t)kv * DK + (e % CH) * 8) : (bf16v8)(bf16)0.0f;
            }
        };
#pragma unroll
        for (int pd = 0; pd < PD; pd++)
            if (lo + pd * BKV < hi) fetch(nxt[pd], lo + pd * BKV);

        f32x16 oacc[NDT];
#pragma unroll
        for (int t = 0; t < NDT; t++) oacc[t] = (f32x16)(0.0f);
        float m_st[4], l_st[4];
#pragma unroll
        for (int i = 0; i < 4; i++) { m_st[i] = FA_NEG_INF; l_st[i] = 0.0f; }

        for (unsigned kb = lo; kb < hi; kb += PD * BKV)
#pragma unroll
        for (int pd = 0; pd < PD; pd++) {
            const unsigned kv0 = kb + pd * BKV;
            if (kv0 >= hi) break;
            __syncthreads();
#pragma unroll
            for (int j = 0; j < TPT; j++) {
                const unsigned e = tid + (unsigned)j * PLOW_THREADS;
                *(bf16v8*)&Ksm[(e / CH) * STRIDE + (e % CH) * 8] = nxt[pd][j];
            }
            __syncthreads();
            if (kv0 + PD * BKV < hi) fetch(nxt[pd], kv0 + PD * BKV);

            f32x16 s[NB];
#pragma unroll
            for (int nb = 0; nb < NB; nb++) s[nb] = (f32x16)(0.0f);
#pragma unroll
            for (int q = 0; q < KPW; q++) {
                const unsigned d0 = mfma_frag_k(lane, (wave * KPW + q) * MFMA_K);
                bf16x8 qf = zero8;
                if (frow < NH) {
                    const bf16v8 qv = ld_lds8(&Qsm[frow * STRIDE + d0]);
                    __builtin_memcpy(&qf, &qv, 16);
                }
#pragma unroll
                for (int nb = 0; nb < NB; nb++) {
                    bf16x8 kf;
                    const bf16v8 kv8 = ld_lds8(&Ksm[(nb * 32 + frow) * STRIDE + d0]);
                    __builtin_memcpy(&kf, &kv8, 16);
                    s[nb] = plow_mfma_bf16_32x32(qf, kf, s[nb]);
                }
            }
#pragma unroll
            for (int nb = 0; nb < NB; nb++)
#pragma unroll
                for (int i = 0; i < 4; i++)
                    Sred[(wave * NH + mfma_acc_m(lane, i)) * BKV + nb * 32 + accn] = s[nb][i];
            __syncthreads();

            /* Every wave reduces the same scores; wave w writes P for keys it owns below. */
            float p[NB][4];
#pragma unroll
            for (int nb = 0; nb < NB; nb++)
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    const unsigned m = mfma_acc_m(lane, i), n = nb * 32 + accn;
                    float acc = 0.0f;
#pragma unroll
                    for (int ww = 0; ww < PLOW_WAVES; ww++) acc += Sred[(ww * NH + m) * BKV + n];
                    p[nb][i] = (kv0 + n < hi) ? acc * FA_SCALE(scale) : FA_NEG_INF;
                }
#pragma unroll
            for (int i = 0; i < 4; i++) {
                float rmax = p[0][i];
#pragma unroll
                for (int nb = 1; nb < NB; nb++) rmax = fmaxf(rmax, p[nb][i]);
                rmax = half_wave_max(rmax);
                const float mnew = fmaxf(m_st[i], rmax);
                const float corr = (m_st[i] == FA_NEG_INF) ? 0.0f : FA_EXP(m_st[i] - mnew);
                float psum = 0.0f;
#pragma unroll
                for (int nb = 0; nb < NB; nb++) {
                    const float pe = (mnew == FA_NEG_INF) ? 0.0f : FA_EXP(p[nb][i] - mnew);
                    p[nb][i] = pe;
                    psum += pe;
                }
                l_st[i] = l_st[i] * corr + half_wave_sum(psum);
                m_st[i] = mnew;
#pragma unroll
                for (int t = 0; t < NDT; t++) oacc[t][i] *= corr;
            }
            if (wave == 0) {
#pragma unroll
                for (int nb = 0; nb < NB; nb++)
#pragma unroll
                    for (int i = 0; i < 4; i++) Psm[mfma_acc_m(lane, i) * BKV + nb * 32 + accn] = f2bf(p[nb][i]);
            }
            __syncthreads();

#pragma unroll
            for (int t = 0; t < NDT; t++) {
#pragma unroll
                for (int ks = 0; ks < BKV; ks += MFMA_K) {
                    const unsigned kk = mfma_frag_k(lane, ks);
                    bf16x8 pf = zero8, vf;
                    if (frow < NH) {
                        const bf16v8 pv = ld_lds8(&Psm[frow * BKV + kk]);
                        __builtin_memcpy(&pf, &pv, 16);
                    }
                    const unsigned col = ncol0 + t * MFMA_N + frow;
#pragma unroll
                    for (int j = 0; j < 8; j++) {
                        const bf16 vv = Ksm[(kk + j) * STRIDE + col];
                        bf16_t vb;
                        __builtin_memcpy(&vb, &vv, 2);
                        vf[j] = vb;
                    }
                    oacc[t] = plow_mfma_bf16_32x32(pf, vf, oacc[t]);
                }
            }
        }

#pragma unroll
        for (int t = 0; t < NDT; t++) {
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const unsigned m = mfma_acc_m(lane, i);
                const unsigned d = ncol0 + t * MFMA_N + accn;
                st_act<float>(&Opart[((size_t)(b * NH + m) * nsplit + sp) * DK + d], oacc[t][i]);
            }
        }
        if (wave == 0 && accn == 0) {
#pragma unroll
            for (int i = 0; i < 4; i++) {
                float* ml = mlpart + ((size_t)(b * NH + mfma_acc_m(lane, i)) * nsplit + sp) * 2;
                st_act<float>(&ml[0], m_st[i]);
                st_act<float>(&ml[1], l_st[i]);
            }
        }
    }
}

/* d_flash_mla_decode_hp8 with the KV tiles moved HBM -> LDS by cp_async16 (no VGPR staging)
 * through an NS-stage ring, NS-1 tiles in flight. Same QK / softmax / PV arithmetic as hp8, so
 * the partials are bit-identical. Each wave moves 4 of a tile's 32 latent rows (one 1 KiB row
 * per wave issue); rows past `hi` re-read row hi-1 (finite, masked to p = 0). */
template <int NS>
__device__ __forceinline__ void hp8_wait_tiles(unsigned after) {
    /* vmcnt counts this wave's loads; 4 per tile. */
    if (NS >= 4 && after >= 2) asm volatile("s_waitcnt vmcnt(8)" ::: "memory");
    else if (NS >= 3 && after >= 1) asm volatile("s_waitcnt vmcnt(4)" ::: "memory");
    else asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
}

template <int DK, int NS>
__device__ void hp8_async(float* __restrict__ Opart, float* __restrict__ mlpart,
                          const bf16* __restrict__ Qabs, const bf16* __restrict__ Ckv,
                          const int* __restrict__ kv_len, unsigned n_batch, unsigned kv_stride,
                          float scale, unsigned nsplit, unsigned slice, unsigned nblk,
                          float* lds_) {
    constexpr int NH = 8;
    constexpr int BKV = 32;
    constexpr int STRIDE = DK + FA_PAD;
    constexpr int NK = DK / MFMA_K;
    constexpr int KPW = NK / PLOW_WAVES;
    constexpr int CPW = DK / PLOW_WAVES;
    constexpr int NDT = CPW / MFMA_N;
    constexpr int CH = DK / 8;
    constexpr int RPW = BKV / PLOW_WAVES; /* latent rows per wave per tile */
    static_assert(DK == 8 * PLOW_WAVE && RPW == 4 && NS >= 2 && NS <= 4, "hp8 async tiling");

    bf16* Kring = (bf16*)lds_;                    /* [NS][BKV][STRIDE] */
    bf16* Qsm = Kring + NS * BKV * STRIDE;        /* [NH][STRIDE] */
    bf16* Psm = Qsm + NH * STRIDE;                /* [WAVES][NH][BKV] */
    float* Sred = (float*)(Psm + PLOW_WAVES * NH * BKV);

    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 63;
    const unsigned wave = tid >> 6;
    const unsigned frow = mfma_frag_row(lane);
    const unsigned accn = mfma_acc_n(lane);
    const unsigned ncol0 = wave * CPW;
    const bf16x8 zero8 = (bf16x8)(bf16_t)0.0f;

    const unsigned n_work = n_batch * nsplit;
    for (unsigned w = slice; w < n_work; w += nblk) {
        const unsigned sp = w % nsplit;
        const unsigned b = w / nsplit;
        const unsigned len = (unsigned)kv_len[b];
        const unsigned per = (len + nsplit - 1) / nsplit;
        const unsigned lo = sp * per < len ? sp * per : len;
        const unsigned hi = lo + per < len ? lo + per : len;
        const auto* cbase = as_glob(Ckv) + (size_t)b * kv_stride * DK;
        const unsigned ntile = (hi - lo + BKV - 1) / BKV;

        __syncthreads();
        if (tid < NH * CH) {
            const unsigned h = tid / CH, c = tid % CH;
            *(bf16v8*)&Qsm[h * STRIDE + c * 8] = ld_glob8(as_glob(Qabs) + ((size_t)b * NH + h) * DK + c * 8);
        }
        auto issue = [&](unsigned t) {
            bf16* st = Kring + (t % NS) * BKV * STRIDE;
#pragma unroll
            for (int r = 0; r < RPW; r++) {
                const unsigned row = wave * RPW + r;
                unsigned kv = lo + t * BKV + row;
                kv = kv < hi ? kv : hi - 1;
                cp_async16(cbase + (size_t)kv * DK + lane * 8, st + row * STRIDE);
            }
        };
#pragma unroll
        for (int t = 0; t < NS - 1; t++)
            if ((unsigned)t < ntile) issue(t);

        f32x16 oacc[NDT];
#pragma unroll
        for (int t = 0; t < NDT; t++) oacc[t] = (f32x16)(0.0f);
        float m_st[4], l_st[4];
#pragma unroll
        for (int i = 0; i < 4; i++) { m_st[i] = FA_NEG_INF; l_st[i] = 0.0f; }

        for (unsigned t = 0; t < ntile; t++) {
            const unsigned kv0 = lo + t * BKV;
            const unsigned left = ntile - 1 - t;
            hp8_wait_tiles<NS>(left < (unsigned)(NS - 2) ? left : (unsigned)(NS - 2));
            __syncthreads();
            if (t + NS - 1 < ntile) issue(t + NS - 1);
            const bf16* Ksm = Kring + (t % NS) * BKV * STRIDE;

            f32x16 s = (f32x16)(0.0f);
#pragma unroll
            for (int q = 0; q < KPW; q++) {
                const unsigned d0 = mfma_frag_k(lane, (wave * KPW + q) * MFMA_K);
                bf16x8 qf = zero8, kf;
                if (frow < NH) {
                    const bf16v8 qv = ld_lds8(&Qsm[frow * STRIDE + d0]);
                    __builtin_memcpy(&qf, &qv, 16);
                }
                const bf16v8 kv8 = ld_lds8(&Ksm[frow * STRIDE + d0]);
                __builtin_memcpy(&kf, &kv8, 16);
                s = plow_mfma_bf16_32x32(qf, kf, s);
            }
#pragma unroll
            for (int i = 0; i < 4; i++)
                Sred[(wave * NH + mfma_acc_m(lane, i)) * BKV + accn] = s[i];
            __syncthreads();

            float p[4];
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const unsigned m = mfma_acc_m(lane, i);
                float acc = 0.0f;
#pragma unroll
                for (int ww = 0; ww < PLOW_WAVES; ww++) acc += Sred[(ww * NH + m) * BKV + accn];
                p[i] = (kv0 + accn < hi) ? acc * FA_SCALE(scale) : FA_NEG_INF;
            }
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const float rmax = half_wave_max(p[i]);
                const float mnew = fmaxf(m_st[i], rmax);
                const float corr = (m_st[i] == FA_NEG_INF) ? 0.0f : FA_EXP(m_st[i] - mnew);
                const float pe = (mnew == FA_NEG_INF) ? 0.0f : FA_EXP(p[i] - mnew);
                l_st[i] = l_st[i] * corr + half_wave_sum(pe);
                m_st[i] = mnew;
                p[i] = pe;
#pragma unroll
                for (int t2 = 0; t2 < NDT; t2++) oacc[t2][i] *= corr;
            }
            bf16* myP = Psm + wave * NH * BKV;
#pragma unroll
            for (int i = 0; i < 4; i++) myP[mfma_acc_m(lane, i) * BKV + accn] = f2bf(p[i]);
            __syncthreads();

#pragma unroll
            for (int t2 = 0; t2 < NDT; t2++) {
#pragma unroll
                for (int ks = 0; ks < BKV; ks += MFMA_K) {
                    const unsigned kk = mfma_frag_k(lane, ks);
                    bf16x8 pf = zero8, vf;
                    if (frow < NH) {
                        const bf16v8 pv = ld_lds8(&myP[frow * BKV + kk]);
                        __builtin_memcpy(&pf, &pv, 16);
                    }
                    const unsigned col = ncol0 + t2 * MFMA_N + frow;
#pragma unroll
                    for (int j = 0; j < 8; j++) {
                        const bf16 vv = Ksm[(kk + j) * STRIDE + col];
                        bf16_t vb;
                        __builtin_memcpy(&vb, &vv, 2);
                        vf[j] = vb;
                    }
                    oacc[t2] = plow_mfma_bf16_32x32(pf, vf, oacc[t2]);
                }
            }
        }

#pragma unroll
        for (int t = 0; t < NDT; t++) {
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const unsigned m = mfma_acc_m(lane, i);
                const unsigned d = ncol0 + t * MFMA_N + accn;
                st_act<float>(&Opart[((size_t)(b * NH + m) * nsplit + sp) * DK + d], oacc[t][i]);
            }
        }
        if (wave == 0 && accn == 0) {
#pragma unroll
            for (int i = 0; i < 4; i++) {
                float* ml = mlpart + ((size_t)(b * NH + mfma_acc_m(lane, i)) * nsplit + sp) * 2;
                st_act<float>(&ml[0], m_st[i]);
                st_act<float>(&ml[1], l_st[i]);
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS) void mla_hp8(
    float* opart, float* mlpart, const bf16* q, const bf16* ckv, const int* kv_len,
    unsigned n_batch, unsigned kv_stride, float scale, unsigned nsplit) {
    extern __shared__ float lds[];
#if defined(HP8_ASYNC)
    hp8_async<512, HP8_ASYNC>(opart, mlpart, q, ckv, kv_len, n_batch, kv_stride, scale, nsplit,
                              blockIdx.x, gridDim.x, lds);
#elif HP8_BKV == 32 && HP8_PD == 1
    d_flash_mla_decode_hp8<512>(opart, mlpart, q, ckv, kv_len, n_batch, kv_stride, scale, nsplit,
                                blockIdx.x, gridDim.x, lds);
#else
    hp8_wide<512, HP8_BKV, HP8_PD>(opart, mlpart, q, ckv, kv_len, n_batch, kv_stride, scale, nsplit,
                           blockIdx.x, gridDim.x, lds);
#endif
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS) void mla_fold_scalar(
    bf16* o, const float* opart, const float* mlpart, const bf16* wuv, unsigned n_batch,
    unsigned n_head, unsigned nsplit) {
    extern __shared__ float lds[];
    d_mla_merge_fold<512, 256>(o, opart, mlpart, wuv, n_batch, n_head, 256, nsplit, blockIdx.x,
                               gridDim.x, lds);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS) void mla_fold_mfma(
    bf16* o, const float* opart, const float* mlpart, const bf16* wuv, unsigned n_batch,
    unsigned n_head, unsigned nsplit) {
    extern __shared__ float lds[];
    d_mla_fold_pf_mfma<1, 256>(o, opart, mlpart, wuv, n_batch, n_head, blockIdx.x, gridDim.x,
                               (bf16*)lds, nsplit);
}

#if PLOW_MLA_FOLD_TB > 1
#define FOLD_TB(N)                                                                                 \
    extern "C" __global__ __launch_bounds__(PLOW_THREADS) void mla_fold_tb##N(                   \
        bf16* o, const float* opart, const float* mlpart, const bf16* wuv, unsigned n_batch,      \
        unsigned n_head, unsigned nsplit) {                                                       \
        extern __shared__ float lds[];                                                            \
        d_mla_merge_fold_tb<512, 256, N>(o, opart, mlpart, wuv, n_batch, n_head, 256, nsplit,     \
                                         blockIdx.x, gridDim.x, lds);                            \
    }
FOLD_TB(2)
FOLD_TB(4)
FOLD_TB(8)
#endif
