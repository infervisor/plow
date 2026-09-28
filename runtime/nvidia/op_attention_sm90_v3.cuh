/* op_attention_sm90_v3.cuh — Hopper flash-attention PREFILL, v3 body (PLOW_NV_FA_V3).
 *
 * Same ABI, work set, masks and log2-domain softmax as d_flash_prefill_sm90 (nsplit == 1, bf16
 * KV, TMA K/V maps required); the caller falls back to that body otherwise. What changes:
 *
 *   - P stays in REGISTERS: the S accumulator fragment of a k16 column slice is exactly the
 *     wgmma A-operand fragment, so P.V is the RS form (no Ps smem round trip, no barrier).
 *   - no redundant QK. HD <= 256: each warpgroup owns 64 DISTINCT query rows of a 128-row item
 *     and the whole head dim (O = 128 f32/lane); both consume the same K/V stages. HD == 512:
 *     both warpgroups own the same 64 rows, each holds half of O, QK is split over the head dim
 *     and the two f32 partial score tiles are exchanged through smem (a + b == b + a, so both
 *     warpgroups see bit-identical S and softmax state).
 *   - K and V arrive by TMA into separate 2-stage rings on mbarriers; the LAST warpgroup to
 *     release a stage refills it, so the warpgroups never meet at a block barrier inside an item.
 *   - a V tail tile past `hi` is zeroed in smem after it lands (P is 0 there, but 0 * NaN is not).
 */
#pragma once

#ifndef PLOW_NV_FA_V3
#define PLOW_NV_FA_V3 0
#endif
/* Rows per TMA box of the packet's GEN_TMAP_KV_PAIR maps. */
#ifndef FA3_BOX
#define FA3_BOX 32
#endif

/* GQA-packed short requests: a request whose gqa x qlen rows fit one 64-row block (a decode row
 * riding a packed prefill launch, a short session suffix) is one item per KV head, its rows
 * m = g * qlen + r stacked over the group's heads, instead of one item per query head that
 * each stream the whole KV. At hd256 the two warpgroups take alternate KV tiles of that block
 * and fold their (m, l, O) through smem at the end. */
#ifndef PLOW_NV_FA_V3_PACK
#define PLOW_NV_FA_V3_PACK 0
#endif
/* Split-KV for launches with fewer items than half the grid (a few riders, a short session
 * suffix): each item's KV tiles are cut into up to FA3_NSK_MAX chunks, one CTA each. A chunk
 * writes its unnormalised (m, l, O) to `ws`; the last chunk to arrive (per-item counter)
 * merges them into O and re-zeroes the counter. The value is the workspace's slot count
 * (devgen `fa_ws`: [slots] u32 counters | [slots][128][2] f32 m,l | [slots][128][512] f32 O);
 * 0 = off. */
#ifndef PLOW_NV_FA_V3_SPLITKV
#define PLOW_NV_FA_V3_SPLITKV 0
#endif
#define FA3_NSK_MAX 16u
#ifndef FA3_SPLIT_MIN_TILES
#define FA3_SPLIT_MIN_TILES 2
#endif
#ifndef FA3_INLINE
#define FA3_INLINE __forceinline__
#endif
#ifndef FA3_ABL
#define FA3_ABL 0
#endif
#define FA3_SPLIT(HD) ((HD) > 256)
#define FA3_BKV(HD) (FA3_SPLIT(HD) ? 32 : 64)
/* smem floats: Q (2 x 64 rows x 256) + 2 stages x (K,V)[BKV][HD] bf16 (+ the hd512 score
 * exchange, 2 buffers x 2 warpgroups x 128 lanes x BKV/2 f32) + 1024 B alignment slack. */
#define FA3_SMEM_FLOATS(HD)                                                                        \
    ((2 * (2 * 64 * 256 + 2 * 2 * FA3_BKV(HD) * (HD)) +                                            \
      (FA3_SPLIT(HD) ? 4 * 2 * 2 * 128 * (FA3_BKV(HD) / 2) : 0) + 1024 + 3) / 4)

/* O[64][64] += P[64][16] . V[16][64]: A (P) from registers, B = V MN-major (trans-b = 1). */
__device__ __forceinline__ void fa3_wgmma_rs_m64n64k16(float* d, const uint32_t* a, uint64_t db) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
        "{%32,%33,%34,%35}, %36, p, 1, 1, 1;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
          "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]),
          "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]),
          "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db));
}

__device__ __forceinline__ uint32_t fa3_pack_bf16(float lo, float hi) {
    __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&v);
}

/* Pin wgmma operand registers so the compiler neither moves their definitions across the
 * async window nor injects its own warpgroup.arrive/wait (CUTLASS warpgroup_fence_operand). */
template <int N> __device__ __forceinline__ void fa3_fence(float* r) {
#pragma unroll
    for (int i = 0; i < N; i++) asm volatile("" : "+f"(r[i])::"memory");
}
template <int N> __device__ __forceinline__ void fa3_fence(uint32_t* r) {
#pragma unroll
    for (int i = 0; i < N; i++) asm volatile("" : "+r"(r[i])::"memory");
}

__device__ __forceinline__ void fa3_bar(int id, int threads) {
    asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(threads) : "memory");
}

/* noinline: its own register allocation instead of the megakernel's merged pressure. */
/* Split-KV (PLOW_NV_FA_V3_SPLITKV) workspace: [SL] u32 counters | [SL][128][2] f32 (m, l) |
 * [SL][128][HD] f32 O, one slot per chunk work item. */
#define FA3_WS_ML(ws) ((ws) + PLOW_NV_FA_V3_SPLITKV)
#define FA3_WS_O(ws) ((ws) + PLOW_NV_FA_V3_SPLITKV * (1 + 128 * 2))

/* After a chunk published its unnormalised state: the last chunk of the item to arrive merges
 * all of them into the bf16 output and re-zeroes the item's counter. Item row ri maps to
 * (query row, head) as the unsplit epilogue does; `scr` is the CTA's idle Q staging area. */
template <int HD, bool SPLIT>
__device__ __noinline__ void fa3_chunk_merge(float* __restrict__ ws, unsigned bitem, unsigned nsk,
                                             unsigned nchunk, bool pk,
                                             __nv_bfloat16* __restrict__ O, unsigned q0,
                                             unsigned sq, unsigned h, unsigned gqa,
                                             unsigned n_head, float* scr) {
    unsigned* const ctr = (unsigned*)ws;
    const float* const wml = FA3_WS_ML(ws);
    const float* const wo = FA3_WS_O(ws);
    __threadfence();
    __syncthreads();
    __shared__ unsigned last;
    if (threadIdx.x == 0) last = atomicAdd(ctr + bitem, 1u) == nchunk - 1u;
    __syncthreads();
    if (!last) return;
    __threadfence();
    const unsigned rows = (pk || SPLIT) ? 64u : 128u;
    const unsigned s0 = bitem * nsk; /* the item's first chunk slot */
    for (unsigned ri = threadIdx.x; ri < rows; ri += PLOW_NV_THREADS) {
        float M = FA_NEG_INF;
        for (unsigned c = 0; c < nchunk; c++)
            M = fmaxf(M, __ldcg(wml + ((size_t)(s0 + c) * 128 + ri) * 2));
        float L = 0.0f;
        for (unsigned c = 0; c < nchunk; c++) {
            const float2 v = __ldcg((const float2*)(wml + ((size_t)(s0 + c) * 128 + ri) * 2));
            const float w = v.x == FA_NEG_INF ? 0.0f : FA_EXP(v.x - M);
            scr[c * 128 + ri] = w;
            L = fmaf(v.y, w, L);
        }
        const float il = L > 0.0f ? 1.0f / L : 0.0f;
        for (unsigned c = 0; c < nchunk; c++) scr[c * 128 + ri] *= il;
    }
    __syncthreads();
    for (unsigned i = threadIdx.x; i < rows * (HD / 4); i += PLOW_NV_THREADS) {
        const unsigned ri = i / (HD / 4), c4 = i % (HD / 4);
        unsigned qr = q0 + ri, hh = h;
        bool ok = qr < sq;
        if (pk) {
            ok = ri < gqa * sq;
            hh = h + ri / sq;
            qr = ri % sq;
        }
        if (!ok) continue;
        float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (unsigned c = 0; c < nchunk; c++) {
            const float w = scr[c * 128 + ri];
            const float4 v = __ldcg((const float4*)(wo + ((size_t)(s0 + c) * 128 + ri) * HD) + c4);
            acc.x = fmaf(v.x, w, acc.x);
            acc.y = fmaf(v.y, w, acc.y);
            acc.z = fmaf(v.z, w, acc.z);
            acc.w = fmaf(v.w, w, acc.w);
        }
        __nv_bfloat162* o2 = (__nv_bfloat162*)(O + ((size_t)qr * n_head + hh) * HD) + 2u * c4;
        o2[0] = __floats2bfloat162_rn(acc.x, acc.y);
        o2[1] = __floats2bfloat162_rn(acc.z, acc.w);
    }
    if (threadIdx.x == 0) ctr[bitem] = 0u;
}

/* KVC: the split-KV instantiation. The unsplit one carries none of its code, so the bulk
 * prefill's register allocation is untouched (the merged body cost hd512 1000 rows +4%). */
template <int HD, bool KVC>
__device__ FA3_INLINE void fa3_v3_body(
    const __nv_bfloat16* __restrict__ Q, __nv_bfloat16* __restrict__ O, unsigned seq_q,
    unsigned seq_kv, unsigned n_head, unsigned n_kv_head, unsigned q_pos0, unsigned window,
    unsigned kv_stride, unsigned kv_mask, float scale, unsigned slice, unsigned nblk, float* lds,
    const int* __restrict__ req, const void* __restrict__ mapkv, float* __restrict__ ws) {
    static_assert(HD == 256 || HD == 512, "v3 covers the Gemma hd256 / hd512 layers");
    static_assert(PLOW_NV_THREADS == 256u, "2 warpgroups");
    constexpr bool SPLIT = FA3_SPLIT(HD);
    constexpr int BKV = FA3_BKV(HD);
    constexpr int ROWS = SPLIT ? 64 : 128;   /* query rows per work item */
    constexpr int NSUB_W = 4;                /* 64-col sub-tiles of this wg's Q / O slice */
    constexpr int NSUB_KV = HD / 64;         /* 64-col sub-tiles of a K/V stage */
    constexpr int QT = 64 * 64;              /* elements per Q sub-tile */
    constexpr int KT = BKV * 64;             /* elements per K/V sub-tile */
    constexpr int NS0 = BKV / 2;             /* S f32 per lane */
    constexpr int KS1 = BKV / 16;            /* P.V k16 steps */
    constexpr int OP_BYTES = NSUB_KV * KT * 2;

    const int tid = threadIdx.x;
    const int wg = tid >> 7;
    const int lt = tid & 127;
    const int w = lt >> 5;
    const int lane = tid & 31;
    const int rA = 16 * w + (lane >> 2);
    const int rB = rA + 8;

    __nv_bfloat16* const base = (__nv_bfloat16*)sm90_align1024(lds);
    __nv_bfloat16* const Qs = base + (size_t)wg * NSUB_W * QT;
    __nv_bfloat16* const Ks = base + 2 * NSUB_W * QT;          /* [2][NSUB_KV][BKV][64] */
    __nv_bfloat16* const Vs = Ks + 2 * NSUB_KV * KT;           /* [2][NSUB_KV][BKV][64] */
    float* const Xs = (float*)(Vs + 2 * NSUB_KV * KT);         /* [2][2][128][NS0] (SPLIT) */

    /* [0..1] K stages, [2..3] V stages */
    __shared__ uint64_t fa3_full[4];
    __shared__ unsigned fa3_rel[4];
    if (tid == 0) {
        for (int s = 0; s < 4; s++) {
            sm90_mbar_init(&fa3_full[s], 1);
            fa3_rel[s] = 0;
        }
    }
    __syncthreads();
    unsigned full_ph = 0;

    const unsigned gqa = n_head / n_kv_head;
    const float lscale = FA_SCALE(scale);

    /* Items of a request: one per KV head when GQA-packed, else (query tile, head). */
    auto req_items = [&](unsigned qlen) -> unsigned {
        if (qlen == 0) return 0u;
        if (PLOW_NV_FA_V3_PACK && qlen * gqa <= 64u) return n_kv_head;
        return ((qlen + ROWS - 1) / ROWS) * n_head;
    };
    unsigned n_work = 0;
    if (req) {
        for (int r = 0; r < req[0]; r++) {
            const int qlen = req[2 + 4 * r];
            if (qlen > 0) n_work += req_items((unsigned)qlen);
        }
    } else {
        n_work = req_items(seq_q);
    }
    /* KV chunks per item (1 = the unsplit body). */
    unsigned nsk = 1;
    if (KVC) {
        nsk = min(min(nblk, (unsigned)PLOW_NV_FA_V3_SPLITKV) / n_work, FA3_NSK_MAX);
        if (nsk < 2u) nsk = 1;
    }
    n_work *= nsk;

    /* Snake over rounds + heaviest query tile first inside a request: causal items grow with
     * the tile index, and pairing round-0 heavy with round-1 light balances the CTAs. */
    for (unsigned round = 0; round * nblk < n_work; round++) {
        const unsigned witem = round * nblk + ((round & 1u) ? nblk - 1u - slice : slice);
        if (witem >= n_work) continue;
        const unsigned bitem = witem / nsk, csp = witem % nsk;
        unsigned h, q0, sq = seq_q, skv = seq_kv, qp0 = q_pos0;
        size_t qoff = 0;
        const void* map = mapkv;
        {
            unsigned rem = bitem;
            if (req) {
                int r = 0, qlen;
                for (;;) {
                    qlen = req[2 + 4 * r];
                    const unsigned nw_r = qlen > 0 ? req_items((unsigned)qlen) : 0u;
                    if (rem < nw_r) break;
                    rem -= nw_r;
                    r++;
                }
                const int rq0 = req[1 + 4 * r], slot = req[3 + 4 * r], kvlen = req[4 + 4 * r];
                sq = (unsigned)qlen;
                skv = (unsigned)kvlen;
                qp0 = (unsigned)(kvlen - qlen);
                qoff = (size_t)rq0 * n_head * HD;
                map = (const void*)((const uint64_t*)mapkv)[slot];
            }
            if (PLOW_NV_FA_V3_PACK && sq * gqa <= 64u) {
                h = rem * gqa; /* first head of the group */
                q0 = 0;
            } else {
                const unsigned ntq = (sq + ROWS - 1) / ROWS;
                h = rem % n_head;
                q0 = (ntq - 1u - rem / n_head) * ROWS;
            }
        }
        const unsigned hkv = h / gqa;
        /* packed: block row m is head h + m / sq, query row m % sq */
        const bool pk = PLOW_NV_FA_V3_PACK && sq * gqa <= 64u;
        const unsigned hi = skv;

        /* KV tile range: union over the item's rows (window floor of the oldest row, causal cap
         * of the newest). */
        const long item_lo_q = (long)qp0 + q0;
        const long item_hi_q = pk ? item_lo_q + (long)sq - 1 : item_lo_q + ROWS - 1;
        unsigned eff_lo = 0;
        if (window) {
            const long wfloor = item_lo_q - (long)window + 1;
            if (wfloor > 0) eff_lo = ((unsigned)wfloor / BKV) * (unsigned)BKV;
        }
        long cap = (long)hi - 1;
        if (item_hi_q < cap) cap = item_hi_q;
        int ntile = (cap >= (long)eff_lo) ? (int)((cap - (long)eff_lo) / BKV) + 1 : 0;
        /* This chunk's tiles; chunks past the item's last tile have no work. */
        unsigned nchunk = 1;
        if (nsk > 1u && ntile > 0) {
            const int per = max(FA3_SPLIT_MIN_TILES, (ntile + (int)nsk - 1) / (int)nsk);
            nchunk = (unsigned)((ntile + per - 1) / per);
            if (csp >= nchunk) continue;
            if (nchunk > 1u) {
                eff_lo += csp * (unsigned)per * BKV;
                ntile = min(per, ntile - (int)csp * per);
            }
        } else if (csp > 0u) {
            continue;
        }
        const bool kvchunk = KVC && nchunk > 1u;

        /* op 0 = K, 1 = V */
        auto issue = [&](int op, int t) {
            const int s = t & 1;
            const unsigned kv0 = eff_lo + (unsigned)t * BKV;
            __nv_bfloat16* dst = (op ? Vs : Ks) + (size_t)s * NSUB_KV * KT;
            const void* m = (const char*)map + 128 * op;
            uint64_t* bar = &fa3_full[2 * op + s];
            sm90_mbar_expect(bar, OP_BYTES);
            const int kvrow = (int)(kv0 & kv_mask);
#pragma unroll
            for (int sub = 0; sub < NSUB_KV; sub++)
#pragma unroll
                for (int hb = 0; hb < BKV / FA3_BOX; hb++)
                    sm90_tma3d(sm90_su32(dst + sub * KT + hb * FA3_BOX * 64), m, sub * 64,
                               kvrow + hb * FA3_BOX, (int)hkv, sm90_su32(bar));
        };
        auto wait_full = [&](int op, int t) {
            const int b = 2 * op + (t & 1);
            sm90_mbar_wait(&fa3_full[b], (int)((full_ph >> b) & 1u));
            full_ph ^= 1u << b;
        };
        /* All four warps of this wg are past their reads of the stage; the second warpgroup to
         * release it issues tile t + 2. */
        auto release = [&](int op, int t) {
            fa90_wg_bar(wg);
            if (lt == 0) {
                const unsigned prev = atomicAdd(&fa3_rel[2 * op + (t & 1)], 1u);
                if ((prev & 1u) && t + 2 < ntile) issue(op, t + 2);
            }
        };

        __syncthreads(); /* previous item's Q / stage reads are complete */
        if (tid == 0) {
            for (int t = 0; t < 2 && t < ntile; t++) {
                issue(0, t);
                issue(1, t);
            }
        }
        /* this wg's Q block: [64 rows][256 cols] */
        {
            const unsigned row0 = q0 + (SPLIT ? 0u : 64u * (unsigned)wg);
            const unsigned col0 = SPLIT ? 256u * (unsigned)wg : 0u;
            const __nv_bfloat16* Qh = Q + qoff + (size_t)h * HD + col0;
            for (int i = lt; i < 64 * NSUB_W * 8; i += 128) {
                const int c = i & 7, sub = (i >> 3) & (NSUB_W - 1), r = i >> 5;
                unsigned qr = row0 + (unsigned)r, hg = 0;
                if (pk) {
                    qr = (unsigned)r % sq;
                    hg = (unsigned)r / sq;
                }
                const bool in = pk ? hg < gqa : qr < sq;
                sm90_cp16(Qs + sub * QT + sm90_swz_off<64, 8>(r, c),
                          Qh + (in ? ((size_t)qr * n_head + hg) * HD + sub * 64 + c * 8 : 0),
                          in ? 16 : 0);
            }
            sm90_cp_commit();
            sm90_cp_wait<0>();
            fa90_async_proxy_fence();
            fa90_wg_bar(wg);
        }

        const long wq_lo = item_lo_q + (SPLIT || pk ? 0 : 64 * wg); /* this wg's oldest row */
        const long wq_hi = pk ? item_hi_q : wq_lo + 63;
        const int qabsA = pk ? (int)item_lo_q + rA % (int)sq : (int)wq_lo + rA;
        const int qabsB = pk ? (int)item_lo_q + rB % (int)sq : (int)wq_lo + rB;
        /* hd256 packed: the warpgroups split the KV tiles (wg w takes t % 2 == w). */
        const bool kvsplit = !SPLIT && pk;

        /* This wg's compute tiles [tb, te]: tiles wholly below its window or above its causal
         * cap are skipped (only the non-split layout has any). */
        int tb = 0, te = ntile - 1;
        if (!SPLIT) {
            while (tb <= te && window &&
                   (long)(eff_lo + (unsigned)tb * BKV) + BKV - 1 < wq_lo - (long)window + 1)
                tb++;
            while (te >= tb && (long)(eff_lo + (unsigned)te * BKV) > wq_hi) te--;
        }
        if (FA3_ABL & 8) te = tb - 1;

        float mA = FA_NEG_INF, lA = 0.0f, mB = FA_NEG_INF, lB = 0.0f;
        float cA = 0.0f, cB = 0.0f;
        float Oacc[NSUB_W][32];
#pragma unroll
        for (int t = 0; t < NSUB_W; t++)
#pragma unroll
            for (int i = 0; i < 32; i++) Oacc[t][i] = 0.0f;
        uint32_t P[KS1][4];

        auto qk_issue = [&](int t, float* S) {
            const __nv_bfloat16* kbuf = Ks + (size_t)(t & 1) * NSUB_KV * KT;
            fa3_fence<NS0>(S);
            sm90_wg_fence();
#pragma unroll
            for (int ks = 0; ks < 16; ks++) {
                const int sub = (SPLIT ? 4 * wg : 0) + (ks >> 2), ko = (ks & 3) * 16;
                if constexpr (BKV == 64)
                    fa90_wgmma_m64n64k16_s(S, sm90_desc(Qs + (ks >> 2) * QT + ko),
                                           sm90_desc(kbuf + sub * KT + ko), ks ? 1 : 0);
                else
                    fa90_wgmma_m64n32k16(S, sm90_desc(Qs + (ks >> 2) * QT + ko),
                                         sm90_desc(kbuf + sub * KT + ko), ks ? 1 : 0);
            }
            sm90_wg_commit();
            fa3_fence<NS0>(S);
        };
        /* Score exchange (split layout), mask, online softmax: S becomes the unnormalised P(t)
         * in place (packed to bf16 once P(t-1).V has drained), the rescale lands in c. */
        auto softmax = [&](int t, float* S) {
            if constexpr (SPLIT) {
                float4* mine = (float4*)(Xs + ((size_t)((t & 1) * 2 + wg) * 128 + lt) * NS0);
                const float4* other =
                    (const float4*)(Xs + ((size_t)((t & 1) * 2 + (wg ^ 1)) * 128 + lt) * NS0);
#pragma unroll
                for (int i = 0; i < NS0 / 4; i++)
                    mine[i] = make_float4(S[4 * i], S[4 * i + 1], S[4 * i + 2], S[4 * i + 3]);
                fa3_bar(3, 256);
#pragma unroll
                for (int i = 0; i < NS0 / 4; i++) {
                    const float4 o = other[i];
                    S[4 * i] += o.x;
                    S[4 * i + 1] += o.y;
                    S[4 * i + 2] += o.z;
                    S[4 * i + 3] += o.w;
                }
            }
            const unsigned kv0 = eff_lo + (unsigned)t * BKV;
            const bool edge = (long)kv0 + BKV - 1 > wq_lo || kv0 + BKV > hi ||
                              (window && (long)kv0 <= wq_hi - (long)window);
            float mxA = FA_NEG_INF, mxB = FA_NEG_INF;
            if (edge) {
#pragma unroll
                for (int nb = 0; nb < BKV / 8; nb++)
#pragma unroll
                    for (int e = 0; e < 2; e++) {
                        const int kv = (int)kv0 + 8 * nb + 2 * (lane & 3) + e;
                        const bool inr = (unsigned)kv < hi;
                        bool okA = inr && kv <= qabsA, okB = inr && kv <= qabsB;
                        if (window) {
                            okA = okA && (unsigned)(qabsA - kv) < window;
                            okB = okB && (unsigned)(qabsB - kv) < window;
                        }
                        const float a = okA ? S[4 * nb + e] * lscale : FA_NEG_INF;
                        const float b = okB ? S[4 * nb + 2 + e] * lscale : FA_NEG_INF;
                        S[4 * nb + e] = a;
                        S[4 * nb + 2 + e] = b;
                        mxA = fmaxf(mxA, a);
                        mxB = fmaxf(mxB, b);
                    }
            } else {
#pragma unroll
                for (int nb = 0; nb < BKV / 8; nb++)
#pragma unroll
                    for (int e = 0; e < 2; e++) {
                        S[4 * nb + e] *= lscale;
                        S[4 * nb + 2 + e] *= lscale;
                        mxA = fmaxf(mxA, S[4 * nb + e]);
                        mxB = fmaxf(mxB, S[4 * nb + 2 + e]);
                    }
            }
            mxA = fmaxf(mxA, __shfl_xor_sync(0xffffffffu, mxA, 1));
            mxA = fmaxf(mxA, __shfl_xor_sync(0xffffffffu, mxA, 2));
            mxB = fmaxf(mxB, __shfl_xor_sync(0xffffffffu, mxB, 1));
            mxB = fmaxf(mxB, __shfl_xor_sync(0xffffffffu, mxB, 2));
            const float mnA = fmaxf(mA, mxA), mnB = fmaxf(mB, mxB);
            cA = (mA == FA_NEG_INF) ? 0.0f : FA_EXP(mA - mnA);
            cB = (mB == FA_NEG_INF) ? 0.0f : FA_EXP(mB - mnB);
            mA = mnA;
            mB = mnB;
            const bool liveA = mnA != FA_NEG_INF, liveB = mnB != FA_NEG_INF;
            float sA = 0.0f, sB = 0.0f;
#pragma unroll
            for (int nb = 0; nb < BKV / 8; nb++) {
                const float pa0 = liveA ? FA_EXP(S[4 * nb + 0] - mnA) : 0.0f;
                const float pa1 = liveA ? FA_EXP(S[4 * nb + 1] - mnA) : 0.0f;
                const float pb0 = liveB ? FA_EXP(S[4 * nb + 2] - mnB) : 0.0f;
                const float pb1 = liveB ? FA_EXP(S[4 * nb + 3] - mnB) : 0.0f;
                sA += pa0 + pa1;
                sB += pb0 + pb1;
                S[4 * nb + 0] = pa0;
                S[4 * nb + 1] = pa1;
                S[4 * nb + 2] = pb0;
                S[4 * nb + 3] = pb1;
            }
            sA += __shfl_xor_sync(0xffffffffu, sA, 1);
            sA += __shfl_xor_sync(0xffffffffu, sA, 2);
            sB += __shfl_xor_sync(0xffffffffu, sB, 1);
            sB += __shfl_xor_sync(0xffffffffu, sB, 2);
            lA = lA * cA + sA;
            lB = lB * cB + sB;
        };
        auto pack = [&](const float* S) {
#pragma unroll
            for (int nb = 0; nb < BKV / 8; nb++) {
                P[nb >> 1][(nb & 1) * 2 + 0] = fa3_pack_bf16(S[4 * nb + 0], S[4 * nb + 1]);
                P[nb >> 1][(nb & 1) * 2 + 1] = fa3_pack_bf16(S[4 * nb + 2], S[4 * nb + 3]);
            }
        };
        auto rescale = [&]() {
#pragma unroll
            for (int nt = 0; nt < NSUB_W; nt++)
#pragma unroll
                for (int nb = 0; nb < 8; nb++)
#pragma unroll
                    for (int e = 0; e < 2; e++) {
                        Oacc[nt][4 * nb + e] *= cA;
                        Oacc[nt][4 * nb + 2 + e] *= cB;
                    }
        };
        /* O += P(t) . V(t) (async; caller commits and waits). */
        auto pv_issue = [&](int t) {
            __nv_bfloat16* const vbuf = Vs + (size_t)(t & 1) * NSUB_KV * KT;
            const unsigned kv0 = eff_lo + (unsigned)t * BKV;
            if (kv0 + BKV > hi) {
                /* V rows past hi were loaded from beyond the live cache: zero them. */
                const int live = (int)(hi - kv0);
                for (int i = lt; i < BKV * NSUB_W * 8; i += 128) {
                    const int c = i & 7, sub = (i >> 3) & (NSUB_W - 1), r = i >> 5;
                    if (r >= live)
                        *(uint4*)(vbuf + ((SPLIT ? 4 * wg : 0) + sub) * KT +
                                  sm90_swz_off<64, 8>(r, c)) = make_uint4(0, 0, 0, 0);
                }
                fa90_async_proxy_fence();
                fa90_wg_bar(wg);
            }
#pragma unroll
            for (int nt = 0; nt < NSUB_W; nt++) fa3_fence<32>(Oacc[nt]);
#pragma unroll
            for (int k = 0; k < KS1; k++) fa3_fence<4>(P[k]);
            sm90_wg_fence();
#pragma unroll
            for (int nt = 0; nt < NSUB_W; nt++) {
                const int g = (SPLIT ? 4 * wg : 0) + nt;
#pragma unroll
                for (int ks = 0; ks < KS1; ks++)
                    fa3_wgmma_rs_m64n64k16(Oacc[nt], P[ks],
                                           sm90_desc(vbuf + g * KT + ks * (16 * 64)));
            }
            sm90_wg_commit();
#pragma unroll
            for (int nt = 0; nt < NSUB_W; nt++) fa3_fence<32>(Oacc[nt]);
#pragma unroll
            for (int k = 0; k < KS1; k++) fa3_fence<4>(P[k]);
        };

        for (int t = 0; t < tb; t++) {
            wait_full(0, t);
            wait_full(1, t);
            release(0, t);
            release(1, t);
        }
        /* Not software-pipelined inside a warpgroup: the two warpgroups already overlap each
         * other's softmax and MMAs, and the pipelined form measured no faster while needing 255
         * registers (it spills inside the interpreter). */
        for (int t = tb; t <= te; t++) {
            if (kvsplit && (t & 1) != wg) {
                wait_full(0, t);
                wait_full(1, t);
                release(0, t);
                release(1, t);
                continue;
            }
            float S[NS0];
            wait_full(0, t);
            qk_issue(t, S);
            sm90_wg_wait<0>();
            fa3_fence<NS0>(S);
            release(0, t);
            softmax(t, S);
            pack(S);
            rescale();
            wait_full(1, t);
            pv_issue(t);
            sm90_wg_wait<0>();
#pragma unroll
            for (int nt = 0; nt < NSUB_W; nt++) fa3_fence<32>(Oacc[nt]);
            release(1, t);
        }
        for (int t = te + 1; t < ntile; t++) {
            wait_full(0, t);
            wait_full(1, t);
            release(0, t);
            release(1, t);
        }

        if (kvsplit) {
            /* Fold wg 1's partial state into wg 0 through the (drained) K/V rings. */
            float* X = (float*)Ks + (size_t)lt * (4 + NSUB_W * 32);
            fa3_bar(3, 256); /* both warpgroups are past their last reads of the rings */
            if (wg == 1) {
                X[0] = mA; X[1] = lA; X[2] = mB; X[3] = lB;
#pragma unroll
                for (int nt = 0; nt < NSUB_W; nt++)
#pragma unroll
                    for (int i = 0; i < 32; i++) X[4 + nt * 32 + i] = Oacc[nt][i];
                fa90_async_proxy_fence();
            }
            fa3_bar(3, 256);
            if (wg == 0) {
                const float MA = fmaxf(mA, X[0]), MB = fmaxf(mB, X[2]);
                const float a0 = mA == FA_NEG_INF ? 0.0f : FA_EXP(mA - MA);
                const float a1 = X[0] == FA_NEG_INF ? 0.0f : FA_EXP(X[0] - MA);
                const float b0 = mB == FA_NEG_INF ? 0.0f : FA_EXP(mB - MB);
                const float b1 = X[2] == FA_NEG_INF ? 0.0f : FA_EXP(X[2] - MB);
                lA = lA * a0 + X[1] * a1;
                lB = lB * b0 + X[3] * b1;
                mA = MA;
                mB = MB;
#pragma unroll
                for (int nt = 0; nt < NSUB_W; nt++)
#pragma unroll
                    for (int nb = 0; nb < 8; nb++)
#pragma unroll
                        for (int e = 0; e < 2; e++) {
                            Oacc[nt][4 * nb + e] =
                                Oacc[nt][4 * nb + e] * a0 + X[4 + nt * 32 + 4 * nb + e] * a1;
                            Oacc[nt][4 * nb + 2 + e] =
                                Oacc[nt][4 * nb + 2 + e] * b0 + X[4 + nt * 32 + 4 * nb + 2 + e] * b1;
                        }
            }
        }
        if constexpr (KVC) if (kvchunk) {
            /* publish this chunk's unnormalised (m, l, O), rows by item row index */
            if (!(kvsplit && wg == 1)) {
                const unsigned riA = (SPLIT || pk ? 0u : 64u * (unsigned)wg) + (unsigned)rA;
                float* const oA = FA3_WS_O(ws) + ((size_t)witem * 128 + riA) * HD;
                float* const oB = oA + 8 * HD;
#pragma unroll
                for (int nt = 0; nt < NSUB_W; nt++)
#pragma unroll
                    for (int nb = 0; nb < 8; nb++) {
                        const int col = (SPLIT ? 256 * wg : 0) + nt * 64 + 8 * nb + 2 * (lane & 3);
                        __stcg((float2*)(oA + col), make_float2(Oacc[nt][4 * nb], Oacc[nt][4 * nb + 1]));
                        __stcg((float2*)(oB + col),
                               make_float2(Oacc[nt][4 * nb + 2], Oacc[nt][4 * nb + 3]));
                    }
                if ((!SPLIT || wg == 0) && (lane & 3) == 0) {
                    float* const ml = FA3_WS_ML(ws) + ((size_t)witem * 128 + riA) * 2;
                    __stcg((float2*)ml, make_float2(mA, lA));
                    __stcg((float2*)(ml + 16), make_float2(mB, lB));
                }
            }
            fa3_chunk_merge<HD, SPLIT>(ws, bitem, nsk, nchunk, pk, O + qoff, q0, sq, h, gqa,
                                       n_head, (float*)base);
            continue;
        }
        const float iA = lA > 0.0f ? 1.0f / lA : 0.0f;
        const float iB = lB > 0.0f ? 1.0f / lB : 0.0f;
        /* Output rows of this lane: (query row, head) for rA and rB. */
        unsigned raA = q0 + (SPLIT ? 0u : 64u * (unsigned)wg) + (unsigned)rA, raB = raA + 8u;
        unsigned hA = h, hB = h;
        bool okA = raA < sq, okB = raB < sq;
        if (pk) {
            hA = h + (unsigned)rA / sq;
            hB = h + (unsigned)rB / sq;
            okA = (unsigned)rA < gqa * sq && !(kvsplit && wg == 1);
            okB = (unsigned)rB < gqa * sq && !(kvsplit && wg == 1);
            raA = (unsigned)rA % sq;
            raB = (unsigned)rB % sq;
        }
#pragma unroll
        for (int nt = 0; nt < NSUB_W; nt++)
#pragma unroll
            for (int nb = 0; nb < 8; nb++) {
                const int col = (SPLIT ? 256 * wg : 0) + nt * 64 + 8 * nb + 2 * (lane & 3);
                if (okA)
                    *(__nv_bfloat162*)(O + qoff + ((size_t)raA * n_head + hA) * HD + col) =
                        __floats2bfloat162_rn(Oacc[nt][4 * nb] * iA, Oacc[nt][4 * nb + 1] * iA);
                if (okB)
                    *(__nv_bfloat162*)(O + qoff + ((size_t)raB * n_head + hB) * HD + col) =
                        __floats2bfloat162_rn(Oacc[nt][4 * nb + 2] * iB,
                                              Oacc[nt][4 * nb + 3] * iB);
            }
    }
    __syncthreads(); /* the static barriers / counters are re-initialised by the next call */
}

template <int HD>
__device__ FA3_INLINE void d_flash_prefill_sm90_v3(
    const __nv_bfloat16* __restrict__ Q, __nv_bfloat16* __restrict__ O, unsigned seq_q,
    unsigned seq_kv, unsigned n_head, unsigned n_kv_head, unsigned q_pos0, unsigned window,
    unsigned kv_stride, unsigned kv_mask, float scale, unsigned slice, unsigned nblk, float* lds,
    const int* __restrict__ req, const void* __restrict__ mapkv, float* __restrict__ ws = nullptr) {
    /* hd512 only: an hd256 layer is sliding (<= 9 tiles) and splitting it measured no faster. */
    if constexpr (PLOW_NV_FA_V3_SPLITKV > 0 && FA3_SPLIT(HD)) {
        if (ws) {
            /* the body's item count, split when it leaves half the grid idle */
            const unsigned gqa = n_head / n_kv_head;
            auto items = [&](unsigned qlen) -> unsigned {
                if (qlen == 0) return 0u;
                if (PLOW_NV_FA_V3_PACK && qlen * gqa <= 64u) return n_kv_head;
                return ((qlen + 63u) / 64u) * n_head;
            };
            unsigned n = 0;
            if (req) {
                for (int r = 0; r < req[0]; r++)
                    if (req[2 + 4 * r] > 0) n += items((unsigned)req[2 + 4 * r]);
            } else {
                n = items(seq_q);
            }
            if (n > 0 && 2u * n <= nblk) {
                fa3_v3_body<HD, true>(Q, O, seq_q, seq_kv, n_head, n_kv_head, q_pos0, window,
                                      kv_stride, kv_mask, scale, slice, nblk, lds, req, mapkv, ws);
                return;
            }
        }
    }
    fa3_v3_body<HD, false>(Q, O, seq_q, seq_kv, n_head, n_kv_head, q_pos0, window, kv_stride,
                           kv_mask, scale, slice, nblk, lds, req, mapkv, ws);
}

