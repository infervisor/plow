#ifndef PLOW_EXPERIMENT_HD512_PC_CUH
#define PLOW_EXPERIMENT_HD512_PC_CUH

/* Exact-shape BF16 HD512/BKV32 prototype: one QK/softmax producer warpgroup feeds two
 * disjoint HD256 PV consumer warpgroups. It is reachable only from the opt-in harness. */

#ifndef PLOW_FA512_PC_PRODUCER_REGS
#define PLOW_FA512_PC_PRODUCER_REGS 32
#endif
#ifndef PLOW_FA512_PC_CONSUMER_REGS
#define PLOW_FA512_PC_CONSUMER_REGS 224
#endif

static_assert(PLOW_FA512_PC_PRODUCER_REGS + 2 * PLOW_FA512_PC_CONSUMER_REGS == 480,
              "the 384-thread entry has a 160-register average budget");

#define FA_SM90_PC512_FLOATS                                                       \
    ((2 * (64 * 512 + 2 * 2 * 32 * 512 + 2 * 64 * 32) +                         \
      2 * 64 * sizeof(float) + 64 * sizeof(float) + 1024 + 3) / 4)

template <int N>
__device__ __forceinline__ void fa90_pc_reg_dec() {
    static_assert(N == 32 || N == 48 || N == 64 || N == 80 || N == 96);
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(N));
}

template <int N>
__device__ __forceinline__ void fa90_pc_reg_inc() {
    static_assert(N == 192 || N == 200 || N == 208 || N == 216 || N == 224);
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(N));
}

template <bool PROD, int HD, int BQ, int BKV>
__device__ __forceinline__ void d_flash_prefill_sm90_pc512(
    const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V, __nv_bfloat16* __restrict__ O,
    unsigned seq_q, unsigned seq_kv, unsigned n_head, unsigned n_kv_head,
    unsigned q_pos0, unsigned window, unsigned nsplit, unsigned kv_stride,
    unsigned kv_mask, float scale, unsigned slice, unsigned nblk, float* lds,
    const void* __restrict__ mapkv) {
    static_assert(HD == 512 && BQ == 64 && BKV == 32,
                  "producer/consumer experiment is HD512/BQ64/BKV32 only");

    if ((seq_q != 4096 && seq_q != 8192) || seq_kv != seq_q || n_head != 16 ||
        n_kv_head != 1 || q_pos0 != 0 || window != 0 || nsplit != 1 ||
        kv_stride < seq_kv || kv_mask != 0xffffffffu || mapkv == nullptr) {
        if (threadIdx.x == 0) __trap();
        return;
    }

    constexpr int NS = 2;
    constexpr int NSUB = HD / 64;
    constexpr int NTW = HD / 128;
    constexpr int KS0 = HD / 16;
    constexpr int KS1 = BKV / 16;
    constexpr int NB0 = BKV / 8;
    constexpr int QT = BQ * 64;
    constexpr int KT = BKV * 64;

    const int tid = (int)threadIdx.x;
    const int lt = tid & 127;
    const int consumer = PROD ? 0 : (tid - 128) >> 7;
    const int warp = (lt >> 5) & 3;
    const int lane = tid & 31;
    const int rA = 16 * warp + (lane >> 2);
    const int rB = rA + 8;

    __nv_bfloat16* const Qs = (__nv_bfloat16*)sm90_align1024(lds);
    __nv_bfloat16* const Ks = Qs + NSUB * QT;
    __nv_bfloat16* const Vs = Ks + NS * NSUB * KT;
    __nv_bfloat16* const Ps = Vs + NS * NSUB * KT;
    float* const corr = (float*)(Ps + NS * BQ * BKV);
    float* const inv = corr + NS * BQ;

    __shared__ uint64_t kv_full[NS];
    __shared__ uint64_t p_full[NS];
    __shared__ uint64_t empty[NS];
    __shared__ uint64_t inv_full;

    const float lscale = FA_SCALE(scale);
    const unsigned n_work = ((seq_q + BQ - 1) / BQ) * n_head;
    for (unsigned witem = slice; witem < n_work; witem += nblk) {
        if constexpr (PROD) {
            if (lt < NS) {
                sm90_mbar_init(kv_full + lt, 1);
                sm90_mbar_init(p_full + lt, 1);
                sm90_mbar_init(empty + lt, 2);
            }
            if (lt == 0) sm90_mbar_init(&inv_full, 1);
            fa90_async_proxy_fence();
        }
        __syncthreads();

        const unsigned h = witem % n_head;
        const unsigned q0 = (witem / n_head) * BQ;
        const __nv_bfloat16* const Qh = Q + (size_t)q0 * n_head * HD + (size_t)h * HD;
        const int qabs_max = (int)(q0 + BQ - 1);
        const long cap = qabs_max < (int)seq_kv ? qabs_max : (long)seq_kv - 1;
        const int ntile = cap >= 0 ? (int)(cap / BKV) + 1 : 0;

        if constexpr (PROD) {
            for (int i = lt; i < BQ * NSUB * 8; i += 128) {
                const int c = i & 7;
                const int sub = (i >> 3) % NSUB;
                const int r = i / (8 * NSUB);
                const bool in = q0 + (unsigned)r < seq_q;
                sm90_cp16(Qs + sub * QT + sm90_swz_off<64, 8>(r, c),
                          Qh + (in ? (size_t)r * n_head * HD +
                                           (size_t)sub * 64 + (size_t)c * 8
                                    : 0),
                          in ? 16 : 0);
            }
            sm90_cp_commit();
            sm90_cp_wait<0>();
            fa90_wg_bar(0);

            auto stage_kv = [&](int tile) {
                const int slot = tile & 1;
                if (tile >= NS)
                    sm90_mbar_wait(empty + slot, ((tile / NS) + 1) & 1);
                if (lt == 0) {
                    sm90_mbar_expect(kv_full + slot, 2 * NSUB * KT * 2);
                    const uint32_t bar = sm90_su32(kv_full + slot);
                    const int kvrow = tile * BKV;
#pragma unroll
                    for (int sub = 0; sub < NSUB; ++sub) {
                        sm90_tma3d(sm90_su32(Ks + slot * NSUB * KT + sub * KT), mapkv,
                                   sub * 64, kvrow, 0, bar);
                        sm90_tma3d(sm90_su32(Vs + slot * NSUB * KT + sub * KT),
                                   (const char*)mapkv + 128, sub * 64, kvrow, 0, bar);
                    }
                }
            };

            float mA = FA_NEG_INF, lA = 0.0f, mB = FA_NEG_INF, lB = 0.0f;
            if (ntile) stage_kv(0);
            for (int tile = 0; tile < ntile; ++tile) {
                const int slot = tile & 1;
                const unsigned kv0 = (unsigned)tile * BKV;
                sm90_mbar_wait(kv_full + slot, (tile / NS) & 1);
                if (tile + 1 < ntile) stage_kv(tile + 1);

                const __nv_bfloat16* const kbuf = Ks + slot * NSUB * KT;
                float S[BKV / 2];
                sm90_wg_fence();
#pragma unroll 1
                for (int ks = 0; ks < KS0; ++ks) {
                    const int sub = ks >> 2;
                    const int ko = (ks & 3) * 16;
                    fa90_wgmma_score<BKV>(S, sm90_desc(Qs + sub * QT + ko),
                                          sm90_desc(kbuf + sub * KT + ko), ks ? 1 : 0);
                }
                sm90_wg_commit();
                sm90_wg_wait<0>();

                float mxA = FA_NEG_INF, mxB = FA_NEG_INF;
#pragma unroll
                for (int nb = 0; nb < NB0; ++nb) {
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        const int kv = (int)kv0 + 8 * nb + 2 * (lane & 3) + e;
                        const bool inr = (unsigned)kv < seq_kv;
                        const bool okA = inr && kv <= (int)q0 + rA;
                        const bool okB = inr && kv <= (int)q0 + rB;
                        const float a = okA ? S[4 * nb + e] * lscale : FA_NEG_INF;
                        const float b = okB ? S[4 * nb + 2 + e] * lscale : FA_NEG_INF;
                        S[4 * nb + e] = a;
                        S[4 * nb + 2 + e] = b;
                        mxA = fmaxf(mxA, a);
                        mxB = fmaxf(mxB, b);
                    }
                }
                mxA = fmaxf(mxA, __shfl_xor_sync(0xffffffffu, mxA, 1));
                mxA = fmaxf(mxA, __shfl_xor_sync(0xffffffffu, mxA, 2));
                mxB = fmaxf(mxB, __shfl_xor_sync(0xffffffffu, mxB, 1));
                mxB = fmaxf(mxB, __shfl_xor_sync(0xffffffffu, mxB, 2));

                const float mnA = fmaxf(mA, mxA), mnB = fmaxf(mB, mxB);
                const float cA = mA == FA_NEG_INF ? 0.0f : FA_EXP(mA - mnA);
                const float cB = mB == FA_NEG_INF ? 0.0f : FA_EXP(mB - mnB);
                mA = mnA;
                mB = mnB;
                const bool liveA = mnA != FA_NEG_INF, liveB = mnB != FA_NEG_INF;
                float sA = 0.0f, sB = 0.0f;
                __nv_bfloat16* const pbuf = Ps + slot * BQ * BKV;
#pragma unroll
                for (int nb = 0; nb < NB0; ++nb) {
                    const float pa0 = liveA ? FA_EXP(S[4 * nb] - mnA) : 0.0f;
                    const float pa1 = liveA ? FA_EXP(S[4 * nb + 1] - mnA) : 0.0f;
                    const float pb0 = liveB ? FA_EXP(S[4 * nb + 2] - mnB) : 0.0f;
                    const float pb1 = liveB ? FA_EXP(S[4 * nb + 3] - mnB) : 0.0f;
                    sA += pa0 + pa1;
                    sB += pb0 + pb1;
                    const int c0 = 8 * nb + 2 * (lane & 3);
                    *(__nv_bfloat162*)(pbuf + fa90_cm_off<BKV>(rA, c0)) =
                        __floats2bfloat162_rn(pa0, pa1);
                    *(__nv_bfloat162*)(pbuf + fa90_cm_off<BKV>(rB, c0)) =
                        __floats2bfloat162_rn(pb0, pb1);
                }
                sA += __shfl_xor_sync(0xffffffffu, sA, 1);
                sA += __shfl_xor_sync(0xffffffffu, sA, 2);
                sB += __shfl_xor_sync(0xffffffffu, sB, 1);
                sB += __shfl_xor_sync(0xffffffffu, sB, 2);
                lA = lA * cA + sA;
                lB = lB * cB + sB;
                if ((lane & 3) == 0) {
                    corr[slot * BQ + rA] = cA;
                    corr[slot * BQ + rB] = cB;
                }
                fa90_async_proxy_fence();
                if (lt == 0) sm90_mbar_arrive(p_full + slot);
            }
            if ((lane & 3) == 0) {
                inv[rA] = lA > 0.0f ? 1.0f / lA : 0.0f;
                inv[rB] = lB > 0.0f ? 1.0f / lB : 0.0f;
            }
            fa90_async_proxy_fence();
            if (lt == 0) sm90_mbar_arrive(&inv_full);
        } else {
            float Oacc[NTW][32];
#pragma unroll
            for (int nt = 0; nt < NTW; ++nt)
#pragma unroll
                for (int i = 0; i < 32; ++i) Oacc[nt][i] = 0.0f;

            for (int tile = 0; tile < ntile; ++tile) {
                const int slot = tile & 1;
                sm90_mbar_wait(p_full + slot, (tile / NS) & 1);
                const float cA = corr[slot * BQ + rA];
                const float cB = corr[slot * BQ + rB];
#pragma unroll
                for (int nt = 0; nt < NTW; ++nt)
#pragma unroll
                    for (int nb = 0; nb < 8; ++nb)
#pragma unroll
                        for (int e = 0; e < 2; ++e) {
                            Oacc[nt][4 * nb + e] *= cA;
                            Oacc[nt][4 * nb + 2 + e] *= cB;
                        }
                fa90_async_proxy_fence();
                const __nv_bfloat16* const pbuf = Ps + slot * BQ * BKV;
                const __nv_bfloat16* const vbuf = Vs + slot * NSUB * KT;
                sm90_wg_fence();
#pragma unroll
                for (int nt = 0; nt < NTW; ++nt) {
                    const int g = consumer * NTW + nt;
#pragma unroll
                    for (int ks = 0; ks < KS1; ++ks)
                        fa90_wgmma_m64n64k16_tb1(
                            Oacc[nt], fa90_desc_ns(pbuf + ks * 128, 128, 16 * BKV),
                            sm90_desc(vbuf + g * KT + ks * (16 * 64)), 1);
                }
                sm90_wg_commit();
                sm90_wg_wait<0>();
                if (lt == 0) sm90_mbar_arrive(empty + slot);
            }

            sm90_mbar_wait(&inv_full, 0);
            const float iA = inv[rA], iB = inv[rB];
#pragma unroll
            for (int nt = 0; nt < NTW; ++nt) {
                const int g = consumer * NTW + nt;
#pragma unroll
                for (int nb = 0; nb < 8; ++nb) {
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        const int hd = g * 64 + 8 * nb + 2 * (lane & 3) + e;
                        const unsigned raA = q0 + (unsigned)rA;
                        const unsigned raB = q0 + (unsigned)rB;
                        if (raA < seq_q)
                            O[(size_t)(raA * n_head + h) * HD + hd] =
                                __float2bfloat16(Oacc[nt][4 * nb + e] * iA);
                        if (raB < seq_q)
                            O[(size_t)(raB * n_head + h) * HD + hd] =
                                __float2bfloat16(Oacc[nt][4 * nb + 2 + e] * iB);
                    }
                }
            }
        }

        __syncthreads();
        if constexpr (PROD) {
            if (lt < NS) {
                sm90_mbar_inval(kv_full + lt);
                sm90_mbar_inval(p_full + lt);
                sm90_mbar_inval(empty + lt);
            }
            if (lt == 0) sm90_mbar_inval(&inv_full);
        }
        __syncthreads();
    }
}

#endif
