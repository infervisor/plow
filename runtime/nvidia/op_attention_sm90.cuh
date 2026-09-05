/* op_attention_sm90.cuh — Hopper (sm_90a) WGMMA flash-attention PREFILL.
 *
 * The sm_90a fork of d_flash_prefill<HD,BQ,BKV>'s hd256 (sliding-layer) arm. Same signature,
 * same numerics contract, same work-item enumeration (including PX-1 varlen `req`) as the shared
 * mma.sync body in op_attention.cuh — only the per-tile math and the smem layout change. hd512
 * (BQ=32) is NOT eligible (wgmma is m64) and keeps the px4 + cp.async.bulk arm untouched.
 *
 * PROVENANCE. Lifted from runtime/nvidia/experiments/wgmma_flash_prefill_probe.cu (12/12 cases
 * vs an f32 CPU oracle, worst relL2 1.77e-3; 2.82x over an oracle-validated FA-2 mma.sync
 * baseline). Its two LOAD-BEARING findings are preserved verbatim:
 *
 *   (a) SOFTMAX STAYS IN-FRAGMENT. A lane owns exactly TWO rows (rA = 16w + lane/4, rB = rA + 8)
 *       in BOTH accumulators, so scale, causal/sliding mask, rowmax, exp, rowsum and the O
 *       rescale all happen in registers; the row reduction is a QUAD butterfly (__shfl_xor 1
 *       then 2 — the 4 lanes of a quad share the two rows and hold disjoint column sets).
 *       Routing S through smem (the shape the mma.sync body uses) measured 0.785x — SLOWER than
 *       mma.sync. Do not reintroduce Ss/m_arr/l_arr/corr_arr here.
 *   (b) V IS MN-MAJOR (trans-b = 1), consumed in its NATURAL [BKV][HD] layout. Transposing V in
 *       smem cost 46% of the probe's runtime.
 *
 * WHAT CHANGES vs the probe, and it is exactly two things:
 *
 *   1. OPERAND LAYOUT: 128-BYTE SWIZZLE (sm90_wgmma.cuh's recipe) instead of the probe's
 *      swizzle-mode-0 core-matrix packing. Swizzle is a PERFORMANCE requirement (the no-swizzle
 *      1024 B row-core stride puts every row-core on one bank => 8-way conflicts on both the
 *      cp.async store and the wgmma operand read). 128B swizzle fixes the innermost box at
 *      64 bf16 = 128 B, so every [rows][HD] tile is stored as HD/64 independent 1024-B-ALIGNED
 *      [rows][64] sub-tiles. Descriptors: LBO = 16 B, SBO = 1024 B, swizzle bits = 1, and a k16
 *      substep advances the START ADDRESS ONLY by +32 B (= +16 bf16).
 *      MN-MAJOR V USES THE SAME STRIDES (LBO=16, SBO=1024) — the stride roles swap only in the
 *      no-swizzle recipe. Measured in runtime/nvidia/experiments/tma_ws_flash_prefill.cu
 *      (selftest 2: the swapped guess lands at relL2 1.007e+00, this one at 6.5e-08).
 *      Ps (the GEMM1 A operand) is written from REGISTERS, its rows are BKV = 32 bf16 = 64 B, so
 *      it cannot be a 128B-swizzle tile and keeps the probe's swizzle-0 core-matrix packing.
 *
 *   2. 256 THREADS = 2 WARPGROUPS, the fixed megakernel block shape (PLOW_NV_THREADS). The probe
 *      is one warpgroup. The two warpgroups SPLIT THE O ACCUMULATOR BY HEAD DIM (warpgroup g owns
 *      hd [g*HD/2, (g+1)*HD/2), i.e. NTW = HD/128 of the n64 output tiles) and REDUNDANTLY
 *      compute the same S = Q.K^T and the same in-fragment softmax. Rationale:
 *        - it is the only split that keeps finding (a): S's rows must be whole inside one
 *          warpgroup's registers, and a k-split or n-split of QK would need a cross-warpgroup
 *          reduction through smem — exactly the 0.785x shape.
 *        - it holds the O accumulator at NTW*32 = 64 f32/lane, the SAME per-thread accumulator
 *          footprint the shipped mma.sync body has. Giving one warpgroup all of O (the literal
 *          probe shape at 256 threads) would double that to 128 f32/lane in a megakernel that is
 *          already at 255 regs / 162 spill bytes.
 *        - the redundancy is 1.5x the MACs (QK twice, P.V once) but flash prefill here runs at a
 *          few percent of wgmma peak — it is softmax/staging bound, not tensor-core bound.
 *      Both warpgroups run the SAME instruction sequence on the SAME inputs, so their m/l/corr
 *      are bit-identical; only warpgroup 0 stores Ps and the (m,l) partials.
 *
 * STAGE 1: cp.async staging only. TMA needs a CUtensorMap through the packet ABI (separate
 * stage). K/V are double-buffered (NS=2) so tile t+1 streams under GEMM0+softmax+GEMM1 of tile t;
 * TWO block barriers per tile is the minimum this schedule admits.
 */
#ifndef PLOW_OP_ATTENTION_SM90_CUH
#define PLOW_OP_ATTENTION_SM90_CUH

#if !defined(PLOW_NV_HOPPER)
#error "op_attention_sm90.cuh is Hopper-only; include it under PLOW_NV_HOPPER"
#endif

#include "sm90_wgmma.cuh"

/* K/V pipeline depth: tile t+1 streams gmem->smem under tile t's GEMM0+softmax+GEMM1. FIXED at 2
 * (the buffer index is a parity flip, not a modulo ring) — this is a layout constant, not a knob. */
#define FA_SM90_NS 2

#ifndef PLOW_NV_FA_SPLIT_OUTER
#define PLOW_NV_FA_SPLIT_OUTER 0
#endif

/* Which <HD,BQ,BKV> instantiations take the wgmma arm. BQ must be the wgmma m64; HD must split
 * into an even number of n64 tiles (one half per warpgroup); BKV=32 is the n32 score tile and the
 * width of the swizzle-0 Ps tile. Today this is exactly d_flash_prefill_mux<256,64,32>, the
 * sliding-layer arm; hd512 is BQ=32 and stays on px4. */
#ifndef PLOW_NV_FA512_WG
#define PLOW_NV_FA512_WG 0
#endif
/* BKV=16 is the hd512 arm (PLOW_NV_FA512_WG, design (a) of the 32k memo): same HD-split /
 * redundant-S structure, score tile m64n16k16, Ps width 16. Smem at <512,64,16> is ~131 KiB
 * (Qs 64 + K/V ring 32 + Ps 2 + align), inside the arena; O acc = 4 n64 tiles = 128 f32. */
#define FA_SM90_WG_ELIGIBLE(HD, BQ, BKV)                                                           \
    ((BQ) == 64 && ((BKV) == 32 || (BKV) == 64 || (PLOW_NV_FA512_WG && (BKV) == 16)) &&            \
     (HD) % 128 == 0)

/* smem floats claimed by the wgmma arm: Qs[HD/64][BQ][64] + NS x (Ks,Vs)[HD/64][BKV][64] bf16 +
 * Ps[BQ][BKV] bf16, plus the 1024 B the swizzle alignment may burn off the arena base. */
#define FA_SM90_PRE_FLOATS(HD, BQ, BKV)                                                             \
    ((2 * ((BQ) * (HD) + 2 * FA_SM90_NS * (BKV) * (HD) + (BQ) * (BKV)) + 1024 + 3) / 4)
/* T30 wgitem: two independent per-warpgroup partitions (each the full single-item claim). */
#define FA_SM90_WGI_FLOATS(HD, BQ, BKV)                                                             \
    ((2 * 2 * ((BQ) * (HD) + 2 * FA_SM90_NS * (BKV) * (HD) + (BQ) * (BKV) + 512) + 2048 + 3) / 4)

/* ---- local wgmma shapes ------------------------------------------------------------------
 * sm90_wgmma.cuh exposes m64n128k16 (64 f32/lane); flash needs n32 for the score tile and n64
 * for the software-tiled O accumulator, which is what keeps HD=256 at 64 f32/lane. Same PTX
 * form, same scale-d predicate convention. */

/* S[64][32] = Q[64][16] . K[32][16]^T — both operands K-major, trans-a = trans-b = 0. */
__device__ __forceinline__ void fa90_wgmma_m64n32k16(float* d, uint64_t da, uint64_t db,
                                                     int scaleD) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %18, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, "
        "%16, %17, p, 1, 1, 0, 0;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
          "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]),
          "+f"(d[14]), "+f"(d[15])
        : "l"(da), "l"(db), "r"(scaleD));
}

/* S[64][16] = Q[64][16] . K[16][16]^T — the BKV=16 (hd512) score tile. */
__device__ __forceinline__ void fa90_wgmma_m64n16k16(float* d, uint64_t da, uint64_t db,
                                                     int scaleD) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %10, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, "
        "%8, %9, p, 1, 1, 0, 0;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
          "+f"(d[7])
        : "l"(da), "l"(db), "r"(scaleD));
}
/* S[64][64] = Q[64][16] . K[64][16]^T — the BKV=64 score tile (T21; both operands K-major). */
__device__ __forceinline__ void fa90_wgmma_m64n64k16_s(float* d, uint64_t da, uint64_t db,
                                                       int scaleD) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
        "%32, %33, p, 1, 1, 0, 0;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
          "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]),
          "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]),
          "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "l"(da), "l"(db), "r"(scaleD));
}

/* Score wgmma for the arm's BKV: n64 (T21 sliding), n32 (shipped sliding) or n16 (hd512). */
template <int BKV>
__device__ __forceinline__ void fa90_wgmma_score(float* d, uint64_t da, uint64_t db, int scaleD) {
    static_assert(BKV == 64 || BKV == 32 || BKV == 16, "score tile is n64, n32 or n16");
    if constexpr (BKV == 64)
        fa90_wgmma_m64n64k16_s(d, da, db, scaleD);
    else if constexpr (BKV == 32)
        fa90_wgmma_m64n32k16(d, da, db, scaleD);
    else
        fa90_wgmma_m64n16k16(d, da, db, scaleD);
}

/* O[64][64] += P[64][16] . V[16][64] — A K-major, B MN-MAJOR (trans-b = 1), i.e. V in its
 * natural [kv][hd] layout, product = A.B with no implicit transpose. Finding (b). */
__device__ __forceinline__ void fa90_wgmma_m64n64k16_tb1(float* d, uint64_t da, uint64_t db,
                                                         int scaleD) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
        "%32, %33, p, 1, 1, 0, 1;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
          "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]),
          "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]),
          "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "l"(da), "l"(db), "r"(scaleD));
}

/* Generic (STS) writes into smem must be published to the ASYNC proxy before wgmma reads them.
 * cp.async lands in the async proxy already, but Ps is stored from registers. */
__device__ __forceinline__ void fa90_async_proxy_fence() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

/* Swizzle-mode-0 descriptor for the register-written Ps tile (rows are BKV bf16 < 128 B, so the
 * 128B-swizzle recipe does not apply). Core-matrix packed, K-inner. */
__device__ __forceinline__ uint64_t fa90_desc_ns(const void* p, uint64_t lbo, uint64_t sbo) {
    uint64_t a = (uint64_t)__cvta_generic_to_shared(p);
    return sm90_desc_enc(a) | (sm90_desc_enc(lbo) << 16) | (sm90_desc_enc(sbo) << 32);
}

/* Element offset of logical (r,c) inside a swizzle-0 core-matrix-packed [R][W] tile (W = BKV).
 * 8 consecutive c stay 16 contiguous bytes, so a bf16x2 register store is one 32-bit STS. */
template <int W> __device__ __forceinline__ int fa90_cm_off(int r, int c) {
    return (r >> 3) * (W * 8) + (c >> 3) * 64 + (r & 7) * 8 + (c & 7);
}

/* ---- T30: WARPGROUP-PER-WORK-ITEM body (PLOW_NV_FA_WGITEM, HD=256/BKV=32 only) -----------
 * The shared body computes S REDUNDANTLY on both warpgroups (1.5x MACs) and pays two
 * block-wide barriers + two full wgmma drains per KV tile. Here each warpgroup owns a WHOLE
 * work item — its own Qs/KV-ring/Ps smem partition, its own mbarriers, wg-scoped named
 * barriers — so the two items' compute/softmax/staging phases interleave and the tensor
 * core never waits on a block-wide rendezvous. O = HD/64 n64 tiles = 128 f32/lane (needs
 * the occ-1 255-reg budget). Per-wg smem at <256,64,32>: Qs 32K + ring 2x(K16+V16)K + Ps 4K
 * ~ 100 KiB, x2 warpgroups ~ 202 KiB. Numerics identical per item (same tile math, same
 * enumeration set) — token identity expected. */
#ifndef PLOW_NV_FA_WGITEM
#define PLOW_NV_FA_WGITEM 0
#endif

__device__ __forceinline__ void fa90_wg_bar(int wg) {
    asm volatile("bar.sync %0, %1;" ::"r"(wg + 1), "r"(128) : "memory");
}

template <int HD, int BQ, int BKV>
__device__ void d_flash_prefill_sm90_wgitem(
    float* __restrict__ Opart, float* __restrict__ mlpart, const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K, const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O, unsigned seq_q, unsigned seq_kv, unsigned n_head,
    unsigned n_kv_head, unsigned q_pos0, unsigned window, unsigned nsplit, unsigned kv_stride,
    unsigned kv_mask, float scale, unsigned slice, unsigned nblk, float* lds,
    const int* __restrict__ req, const void* __restrict__ mapkv) {
    static_assert(HD == 256 && BQ == 64 && BKV == 32, "wgitem body is the <256,64,32> shape");
    constexpr int NSUB = HD / 64;
    constexpr int NTO = HD / 64; /* n64 O tiles per wg — the FULL head dim */
    constexpr int KS0 = HD / 16;
    constexpr int KS1 = BKV / 16;
    constexpr int NB0 = BKV / 8;
    constexpr int QT = BQ * 64;
    constexpr int KT = BKV * 64;
    /* per-wg smem partition (elements), 1024B-aligned per wg */
    constexpr int PERWG =
        NSUB * QT + 2 * FA_SM90_NS * NSUB * KT + BQ * BKV + 512 /* align slack */;

    const int tid = threadIdx.x;
    const int wg = tid >> 7;
    const int lt = tid & 127; /* thread within the warpgroup */
    const int w = (lt >> 5);
    const int lane = tid & 31;
    const int rA = 16 * w + (lane >> 2);
    const int rB = rA + 8;

    __nv_bfloat16* const base0 = (__nv_bfloat16*)sm90_align1024(lds);
    __nv_bfloat16* const Qs = (__nv_bfloat16*)sm90_align1024(base0 + (size_t)wg * PERWG);
    __nv_bfloat16* const Ks = Qs + NSUB * QT;
    __nv_bfloat16* const Vs = Ks + FA_SM90_NS * NSUB * KT;
    __nv_bfloat16* const Ps = Vs + FA_SM90_NS * NSUB * KT;

    __shared__ uint64_t fa90w_bar[2][FA_SM90_NS];
    unsigned fa90_ph[FA_SM90_NS] = {0, 0};
    bool fa90_tma[FA_SM90_NS] = {false, false};
    if (mapkv && lt == 0) {
        sm90_mbar_init(&fa90w_bar[wg][0], 1);
        sm90_mbar_init(&fa90w_bar[wg][1], 1);
    }
    if (mapkv) fa90_wg_bar(wg);

    const unsigned gqa = n_head / n_kv_head;
    const float lscale = FA_SCALE(scale);

    unsigned n_work;
    if (req) {
        n_work = 0;
        for (int r = 0; r < req[0]; r++) {
            const int qlen = req[2 + 4 * r];
            if (qlen > 0) n_work += (unsigned)((qlen + BQ - 1) / BQ) * n_head;
        }
    } else {
        n_work = ((seq_q + BQ - 1) / BQ) * n_head * nsplit;
    }

    for (unsigned witem = slice * 2u + (unsigned)wg; witem < n_work; witem += nblk * 2u) {
        unsigned sp, h, q0, sq = seq_q, skv = seq_kv, qp0 = q_pos0, ns = nsplit;
        size_t qoff = 0, kvoff = 0;
        if (req) {
            unsigned rem = witem;
            int r = 0, qlen;
            for (;;) {
                qlen = req[2 + 4 * r];
                const unsigned nw_r = (qlen > 0) ? (unsigned)((qlen + BQ - 1) / BQ) * n_head : 0u;
                if (rem < nw_r) break;
                rem -= nw_r;
                r++;
            }
            const int rq0 = req[1 + 4 * r], slot = req[3 + 4 * r], kvlen = req[4 + 4 * r];
            sp = 0;
            ns = 1;
            h = rem % n_head;
            q0 = (rem / n_head) * BQ;
            sq = (unsigned)qlen;
            skv = (unsigned)kvlen;
            qp0 = (unsigned)(kvlen - qlen);
            qoff = (size_t)rq0 * n_head * HD;
            kvoff = (size_t)slot * n_kv_head * (size_t)kv_stride * HD;
        } else {
            sp = witem % nsplit;
            h = (witem / nsplit) % n_head;
            q0 = (witem / (nsplit * n_head)) * BQ;
        }
        const unsigned hkv = h / gqa;

        const unsigned per = (skv + ns - 1) / ns;
        const unsigned lo = sp * per;
        const unsigned hi = (lo + per < skv) ? (lo + per) : skv;

        const __nv_bfloat16* Qh = Q + qoff + (size_t)q0 * n_head * HD + (size_t)h * HD;
        const __nv_bfloat16* Kb = K + kvoff + (size_t)hkv * kv_stride * HD;
        const __nv_bfloat16* Vb = V + kvoff + (size_t)hkv * kv_stride * HD;

        const int qabs_max = (int)(qp0 + q0 + BQ - 1);
        unsigned eff_lo = lo;
        if (window) {
            const long wfloor = (long)qp0 + q0 - (long)window + 1;
            if (wfloor > (long)lo) eff_lo = ((unsigned)wfloor / BKV) * (unsigned)BKV;
        }
        long cap = (long)hi - 1;
        if ((long)qabs_max < cap) cap = (long)qabs_max;
        const int ntile = (cap >= (long)eff_lo) ? (int)((cap - (long)eff_lo) / BKV) + 1 : 0;

        const bool use_tma = mapkv != nullptr && kvoff == 0;
        auto stageKV = [&](unsigned kv0, int buf) {
            __nv_bfloat16* kd = Ks + (size_t)buf * NSUB * KT;
            __nv_bfloat16* vd = Vs + (size_t)buf * NSUB * KT;
            const bool full = use_tma && (kv0 + (unsigned)BKV <= hi);
            fa90_tma[buf] = full;
            if (full) {
                if (lt == 0) {
                    sm90_mbar_expect(&fa90w_bar[wg][buf], 2 * NSUB * KT * 2);
                    const uint32_t bar = sm90_su32(&fa90w_bar[wg][buf]);
                    const int kvrow = (int)(kv0 & kv_mask);
#pragma unroll
                    for (int sub = 0; sub < NSUB; sub++) {
                        sm90_tma3d(sm90_su32(kd + sub * KT), mapkv, sub * 64, kvrow, (int)hkv,
                                   bar);
                        sm90_tma3d(sm90_su32(vd + sub * KT), (const char*)mapkv + 128, sub * 64,
                                   kvrow, (int)hkv, bar);
                    }
                }
                return;
            }
            for (int i = lt; i < BKV * NSUB * 8; i += 128) {
                const int c = i & 7, sub = (i >> 3) % NSUB, r = i / (8 * NSUB);
                const unsigned kv = kv0 + (unsigned)r;
                const bool in = (kv < hi);
                const size_t row = (size_t)(kv & kv_mask);
                const int soff = sub * KT + sm90_swz_off<64, 8>(r, c);
                const size_t goff = in ? row * HD + (size_t)sub * 64 + (size_t)c * 8 : 0;
                sm90_cp16(kd + soff, Kb + goff, in ? 16 : 0);
                sm90_cp16(vd + soff, Vb + goff, in ? 16 : 0);
            }
            sm90_cp_commit();
        };
        auto waitKV = [&](int buf) {
            sm90_cp_wait<0>();
            if (fa90_tma[buf]) {
                sm90_mbar_wait(&fa90w_bar[wg][buf], (int)(fa90_ph[buf] & 1u));
                fa90_ph[buf]++;
                fa90_tma[buf] = false;
            }
        };

        fa90_wg_bar(wg); /* previous item's Qs/Ks/Vs/Ps reads complete before restaging */

        for (int i = lt; i < BQ * NSUB * 8; i += 128) {
            const int c = i & 7, sub = (i >> 3) % NSUB, r = i / (8 * NSUB);
            const bool in = (q0 + (unsigned)r < sq);
            sm90_cp16(Qs + sub * QT + sm90_swz_off<64, 8>(r, c),
                      Qh + (in ? (size_t)r * n_head * HD + (size_t)sub * 64 + (size_t)c * 8 : 0),
                      in ? 16 : 0);
        }
        sm90_cp_commit();
        if (ntile > 0) stageKV(eff_lo, 0);
        waitKV(0);

        float mA = FA_NEG_INF, lA = 0.0f, mB = FA_NEG_INF, lB = 0.0f;
        float Oacc[NTO][32];
#pragma unroll
        for (int t = 0; t < NTO; t++)
#pragma unroll
            for (int i = 0; i < 32; i++) Oacc[t][i] = 0.0f;

        const int qabsA = (int)(qp0 + q0) + rA, qabsB = (int)(qp0 + q0) + rB;
        int sb = 0;

        for (int t = 0; t < ntile; t++) {
            const unsigned kv0 = eff_lo + (unsigned)t * BKV;
            fa90_wg_bar(wg); /* (A) tile t visible wg-wide; tile t-1's buffer + Ps free */
            fa90_async_proxy_fence();
            if (t + 1 < ntile) stageKV(kv0 + BKV, sb ^ 1);

            const __nv_bfloat16* kbuf = Ks + (size_t)sb * NSUB * KT;
            float S[BKV / 2];
            sm90_wg_fence();
#pragma unroll 1
            for (int ks = 0; ks < KS0; ks++) {
                const int sub = ks >> 2, ko = (ks & 3) * 16;
                fa90_wgmma_score<BKV>(S, sm90_desc(Qs + sub * QT + ko),
                                      sm90_desc(kbuf + sub * KT + ko), ks ? 1 : 0);
            }
            sm90_wg_commit();
            sm90_wg_wait<0>();

            float mxA = FA_NEG_INF, mxB = FA_NEG_INF;
#pragma unroll
            for (int nb = 0; nb < NB0; nb++)
#pragma unroll
                for (int e = 0; e < 2; e++) {
                    const int kv = (int)kv0 + 8 * nb + 2 * (lane & 3) + e;
                    const bool inr = ((unsigned)kv < hi);
                    bool okA = inr && (kv <= qabsA), okB = inr && (kv <= qabsB);
                    if (window) {
                        okA = okA && ((unsigned)(qabsA - kv) < window);
                        okB = okB && ((unsigned)(qabsB - kv) < window);
                    }
                    const float a = okA ? S[4 * nb + e] * lscale : FA_NEG_INF;
                    const float b = okB ? S[4 * nb + 2 + e] * lscale : FA_NEG_INF;
                    S[4 * nb + e] = a;
                    S[4 * nb + 2 + e] = b;
                    mxA = fmaxf(mxA, a);
                    mxB = fmaxf(mxB, b);
                }
            mxA = fmaxf(mxA, __shfl_xor_sync(0xffffffffu, mxA, 1));
            mxA = fmaxf(mxA, __shfl_xor_sync(0xffffffffu, mxA, 2));
            mxB = fmaxf(mxB, __shfl_xor_sync(0xffffffffu, mxB, 1));
            mxB = fmaxf(mxB, __shfl_xor_sync(0xffffffffu, mxB, 2));

            const float mnA = fmaxf(mA, mxA), mnB = fmaxf(mB, mxB);
            const float cA = (mA == FA_NEG_INF) ? 0.0f : FA_EXP(mA - mnA);
            const float cB = (mB == FA_NEG_INF) ? 0.0f : FA_EXP(mB - mnB);
            mA = mnA;
            mB = mnB;

            const bool liveA = (mnA != FA_NEG_INF), liveB = (mnB != FA_NEG_INF);
            float sA = 0.0f, sB = 0.0f;
#pragma unroll
            for (int nb = 0; nb < NB0; nb++) {
                const float pa0 = liveA ? FA_EXP(S[4 * nb + 0] - mnA) : 0.0f;
                const float pa1 = liveA ? FA_EXP(S[4 * nb + 1] - mnA) : 0.0f;
                const float pb0 = liveB ? FA_EXP(S[4 * nb + 2] - mnB) : 0.0f;
                const float pb1 = liveB ? FA_EXP(S[4 * nb + 3] - mnB) : 0.0f;
                sA += pa0 + pa1;
                sB += pb0 + pb1;
                const int c0 = 8 * nb + 2 * (lane & 3);
                *(__nv_bfloat162*)(Ps + fa90_cm_off<BKV>(rA, c0)) =
                    __floats2bfloat162_rn(pa0, pa1);
                *(__nv_bfloat162*)(Ps + fa90_cm_off<BKV>(rB, c0)) =
                    __floats2bfloat162_rn(pb0, pb1);
            }
            sA += __shfl_xor_sync(0xffffffffu, sA, 1);
            sA += __shfl_xor_sync(0xffffffffu, sA, 2);
            sB += __shfl_xor_sync(0xffffffffu, sB, 1);
            sB += __shfl_xor_sync(0xffffffffu, sB, 2);
            lA = lA * cA + sA;
            lB = lB * cB + sB;

#pragma unroll
            for (int nt = 0; nt < NTO; nt++)
#pragma unroll
                for (int nb = 0; nb < 8; nb++)
#pragma unroll
                    for (int e = 0; e < 2; e++) {
                        Oacc[nt][4 * nb + e] *= cA;
                        Oacc[nt][4 * nb + 2 + e] *= cB;
                    }

            fa90_wg_bar(wg); /* (B) Ps published to the warpgroup */
            fa90_async_proxy_fence();

            const __nv_bfloat16* vbuf = Vs + (size_t)sb * NSUB * KT;
            sm90_wg_fence();
#pragma unroll
            for (int nt = 0; nt < NTO; nt++) {
#pragma unroll 1
                for (int ks = 0; ks < KS1; ks++)
                    fa90_wgmma_m64n64k16_tb1(Oacc[nt],
                                             fa90_desc_ns(Ps + ks * 128, 128, 16 * BKV),
                                             sm90_desc(vbuf + nt * KT + ks * (16 * 64)), 1);
            }
            sm90_wg_commit();
            sm90_wg_wait<0>();
            waitKV(sb ^ 1);
            sb ^= 1;
        }

        const float iA = (lA > 0.0f) ? 1.0f / lA : 0.0f;
        const float iB = (lB > 0.0f) ? 1.0f / lB : 0.0f;
#pragma unroll
        for (int nt = 0; nt < NTO; nt++) {
#pragma unroll
            for (int nb = 0; nb < 8; nb++)
#pragma unroll
                for (int e = 0; e < 2; e++) {
                    const int hd = nt * 64 + 8 * nb + 2 * (lane & 3) + e;
                    const unsigned raA = q0 + (unsigned)rA, raB = q0 + (unsigned)rB;
                    const float va = Oacc[nt][4 * nb + e], vb = Oacc[nt][4 * nb + 2 + e];
                    if (ns > 1) {
                        if (raA < sq)
                            Opart[((size_t)(raA * n_head + h) * ns + sp) * HD + hd] = va;
                        if (raB < sq)
                            Opart[((size_t)(raB * n_head + h) * ns + sp) * HD + hd] = vb;
                    } else {
                        if (raA < sq)
                            O[qoff + (size_t)(raA * n_head + h) * HD + hd] =
                                __float2bfloat16(va * iA);
                        if (raB < sq)
                            O[qoff + (size_t)(raB * n_head + h) * HD + hd] =
                                __float2bfloat16(vb * iB);
                    }
                }
        }
        if (ns > 1 && (lane & 3) == 0) {
            const unsigned raA = q0 + (unsigned)rA, raB = q0 + (unsigned)rB;
            if (raA < sq) {
                float* ml = mlpart + ((size_t)(raA * n_head + h) * ns + sp) * 2;
                ml[0] = mA;
                ml[1] = lA;
            }
            if (raB < sq) {
                float* ml = mlpart + ((size_t)(raB * n_head + h) * ns + sp) * 2;
                ml[0] = mB;
                ml[1] = lB;
            }
        }
    }
}

/* ============================ the kernel body ============================================== */
template <int HD, int BQ, int BKV>
__device__ void d_flash_prefill_sm90(float* __restrict__ Opart, float* __restrict__ mlpart,
                                     const __nv_bfloat16* __restrict__ Q,
                                     const __nv_bfloat16* __restrict__ K,
                                     const __nv_bfloat16* __restrict__ V,
                                     __nv_bfloat16* __restrict__ O, unsigned seq_q,
                                     unsigned seq_kv, unsigned n_head, unsigned n_kv_head,
                                     unsigned q_pos0, unsigned window, unsigned nsplit,
                                     unsigned kv_stride, unsigned kv_mask, float scale,
                                     unsigned slice, unsigned nblk, float* lds,
                                     const int* __restrict__ req,
                                     const void* __restrict__ mapkv = nullptr) {
#if PLOW_NV_FA_WGITEM
    if constexpr (HD == 256 && BKV == 32) {
        d_flash_prefill_sm90_wgitem<HD, BQ, BKV>(Opart, mlpart, Q, K, V, O, seq_q, seq_kv,
                                                 n_head, n_kv_head, q_pos0, window, nsplit,
                                                 kv_stride, kv_mask, scale, slice, nblk, lds,
                                                 req, mapkv);
        return;
    }
#endif
    static_assert(BQ == 64, "wgmma is m64: one warpgroup owns the whole q-tile");
    static_assert(BKV == 64 || BKV == 32 || BKV == 16, "score tile is n64/n32/n16");
    static_assert(HD % 128 == 0, "HD splits into an even number of n64 tiles, one half per wg");
    static_assert(PLOW_NV_THREADS == 256u, "2 warpgroups");

    constexpr int NSUB = HD / 64;      /* 128B-swizzle sub-tiles per [rows][HD] operand */
    constexpr int NTW = HD / 128;      /* n64 O tiles per warpgroup                     */
    constexpr int KS0 = HD / 16;       /* GEMM0 k16 steps (contract over HD)            */
    constexpr int KS1 = BKV / 16;      /* GEMM1 k16 steps (contract over BKV)           */
    constexpr int NB0 = BKV / 8;       /* n8 blocks in the score accumulator            */
    constexpr int QT = BQ * 64;        /* elements per Q sub-tile                       */
    constexpr int KT = BKV * 64;       /* elements per K/V sub-tile                     */

    /* --- smem: every 128B-swizzle tile must be 1024 B aligned or the hardware address swizzle
     * disagrees with our store-side XOR and the results are silently WRONG (it does not fault).
     * Sub-tiles are 8 KiB (Q) / 4 KiB (K,V), so aligning the base aligns all of them. --- */
    __nv_bfloat16* const Qs = (__nv_bfloat16*)sm90_align1024(lds);
    __nv_bfloat16* const Ks = Qs + NSUB * QT;                       /* [NS][NSUB][BKV][64] */
    __nv_bfloat16* const Vs = Ks + FA_SM90_NS * NSUB * KT;          /* [NS][NSUB][BKV][64] */
    __nv_bfloat16* const Ps = Vs + FA_SM90_NS * NSUB * KT;          /* [BQ][BKV] swizzle-0 */

    const int tid = threadIdx.x;
    const int wg = tid >> 7;               /* warpgroup 0/1: owns hd [wg*HD/2, +HD/2)      */
    const int w = (tid >> 5) & 3;          /* warp within the warpgroup                    */
    const int lane = tid & 31;
    const int rA = 16 * w + (lane >> 2);   /* the two accumulator rows this lane owns      */
    const int rB = rA + 8;

    /* TMA K/V staging (PLOW_NV_TMA_GEMM path): `mapkv` is the GEN_TMAP_KV_PAIR blob — K's
     * rank-3 map {hd, ring, n_kv_head} at +0, V's at +128, box {64, BKV, 1}. FULL tiles go
     * through the copy engine (8 issues replace ~512 cp.async); the (rare) PARTIAL tail
     * tile keeps cp.async so its V rows past `hi` stay ZEROED (stale smem may hold NaN and
     * 0*NaN != 0 in the mma — the same rule the px4 TMA arm documents). mbarrier phases run
     * continuously across tiles and work items; static smem, so no arena claim and no inval
     * (the barriers are never reused as plain data). */
    __shared__ uint64_t fa90_bar[FA_SM90_NS];
    unsigned fa90_ph[FA_SM90_NS] = {0, 0};
    bool fa90_tma[FA_SM90_NS] = {false, false};
    if (mapkv && tid == 0) {
        sm90_mbar_init(&fa90_bar[0], 1);
        sm90_mbar_init(&fa90_bar[1], 1);
    }
    if (mapkv) __syncthreads();

    const unsigned gqa = n_head / n_kv_head;
    const float lscale = FA_SCALE(scale);

    /* Work-item count — identical enumeration to the shared body (PX-1 varlen included). */
    unsigned n_work;
    if (req) {
        n_work = 0;
        for (int r = 0; r < req[0]; r++) {
            const int qlen = req[2 + 4 * r];
            if (qlen > 0) n_work += (unsigned)((qlen + BQ - 1) / BQ) * n_head;
        }
    } else {
        n_work = ((seq_q + BQ - 1) / BQ) * n_head * nsplit;
    }

    for (unsigned witem = slice; witem < n_work; witem += nblk) {
        unsigned sp, h, q0, sq = seq_q, skv = seq_kv, qp0 = q_pos0, ns = nsplit;
        size_t qoff = 0, kvoff = 0;
        if (req) {
            unsigned rem = witem;
            int r = 0, qlen;
            for (;;) {
                qlen = req[2 + 4 * r];
                const unsigned nw_r = (qlen > 0) ? (unsigned)((qlen + BQ - 1) / BQ) * n_head : 0u;
                if (rem < nw_r) break;
                rem -= nw_r;
                r++;
            }
            const int rq0 = req[1 + 4 * r], slot = req[3 + 4 * r], kvlen = req[4 + 4 * r];
            sp = 0;
            ns = 1;
            h = rem % n_head;
            q0 = (rem / n_head) * BQ;
            sq = (unsigned)qlen;
            skv = (unsigned)kvlen;
            qp0 = (unsigned)(kvlen - qlen);
            qoff = (size_t)rq0 * n_head * HD;
            kvoff = (size_t)slot * n_kv_head * (size_t)kv_stride * HD;
        } else {
#if PLOW_NV_FA_SPLIT_OUTER
            if constexpr (HD == 256 && BQ == 64 && BKV == 32) {
                const unsigned head_tiles = ((seq_q + BQ - 1) / BQ) * n_head;
                sp = witem / head_tiles;
                const unsigned qh = witem % head_tiles;
                h = qh % n_head;
                q0 = (qh / n_head) * BQ;
            } else
#endif
            {
                sp = witem % nsplit;
                h = (witem / nsplit) % n_head;
                q0 = (witem / (nsplit * n_head)) * BQ;
            }
        }
        const unsigned hkv = h / gqa;

        const unsigned per = (skv + ns - 1) / ns;
        const unsigned lo = sp * per;
        const unsigned hi = (lo + per < skv) ? (lo + per) : skv;

        const __nv_bfloat16* Qh = Q + qoff + (size_t)q0 * n_head * HD + (size_t)h * HD;
        const __nv_bfloat16* Kb = K + kvoff + (size_t)hkv * kv_stride * HD;
        const __nv_bfloat16* Vb = V + kvoff + (size_t)hkv * kv_stride * HD;

        /* Tile enumeration — byte-for-byte the shared body's: sliding window floor, causal cap
         * on the newest query in the tile. NO per-q-row causal tile skip is added here (it would
         * change the flop count; separate commit). */
        const int qabs_max = (int)(qp0 + q0 + BQ - 1);
        unsigned eff_lo = lo;
        if (window) {
            const long wfloor = (long)qp0 + q0 - (long)window + 1;
            if (wfloor > (long)lo) eff_lo = ((unsigned)wfloor / BKV) * (unsigned)BKV;
        }
        long cap = (long)hi - 1;
        if ((long)qabs_max < cap) cap = (long)qabs_max;
        const int ntile = (cap >= (long)eff_lo) ? (int)((cap - (long)eff_lo) / BKV) + 1 : 0;

        /* --- staging. cp.async writes 16 B (8 bf16) straight into the 128B-swizzled sub-tile;
         * out-of-range rows pass src-size 0, which zero-fills (so a masked P never multiplies
         * stale V, and QK of a pad row is 0). --- */
        /* TMA eligibility per work item: the pair maps address the base KV tensor; a
         * nonzero batch-slot offset (PX-1 varlen, slot>0) would need slot as a 4th
         * coordinate, so those items keep cp.async. */
        /* The kv-pair map's box is 32 rows, so the staging loop below walks a tile in BKV/32
         * steps: at BKV<32 it runs ZERO times AFTER the barrier is armed for the full byte
         * count, and waitKV() then spins forever (reachable via the documented
         * -DPLOW_NV_FA512_WG=1 -DPLOW_NV_FA_PX4=0 build). BKV is a template constant, so this
         * term folds away; a sub-32 BKV simply keeps the cp.async path below. */
        const bool use_tma = mapkv != nullptr && kvoff == 0 && (BKV % 32 == 0);
        auto stageKV = [&](unsigned kv0, int buf) {
            __nv_bfloat16* kd = Ks + (size_t)buf * NSUB * KT;
            __nv_bfloat16* vd = Vs + (size_t)buf * NSUB * KT;
            const bool full = use_tma && (kv0 + (unsigned)BKV <= hi);
            fa90_tma[buf] = full;
            if (full) {
                if (tid == 0) {
                    sm90_mbar_expect(&fa90_bar[buf], 2 * NSUB * KT * 2);
                    const uint32_t bar = sm90_su32(&fa90_bar[buf]);
                    const int kvrow = (int)(kv0 & kv_mask); /* BKV | ring: no wrap inside a tile */
#pragma unroll
                    for (int sub = 0; sub < NSUB; sub++)
#pragma unroll
                        for (int hb = 0; hb < BKV / 32; hb++) { /* kv-pair map box is 32 rows */
                            sm90_tma3d(sm90_su32(kd + sub * KT + hb * 32 * 64), mapkv, sub * 64,
                                       kvrow + hb * 32, (int)hkv, bar);
                            sm90_tma3d(sm90_su32(vd + sub * KT + hb * 32 * 64),
                                       (const char*)mapkv + 128, sub * 64, kvrow + hb * 32,
                                       (int)hkv, bar);
                        }
                }
                return;
            }
            for (int i = tid; i < BKV * NSUB * 8; i += (int)PLOW_NV_THREADS) {
                /* 32 consecutive threads sweep one KV row's HD*2 contiguous bytes. */
                const int c = i & 7, sub = (i >> 3) % NSUB, r = i / (8 * NSUB);
                const unsigned kv = kv0 + (unsigned)r;
                const bool in = (kv < hi);
                const size_t row = (size_t)(kv & kv_mask);
                const int soff = sub * KT + sm90_swz_off<64, 8>(r, c);
                const size_t goff = in ? row * HD + (size_t)sub * 64 + (size_t)c * 8 : 0;
                sm90_cp16(kd + soff, Kb + goff, in ? 16 : 0);
                sm90_cp16(vd + soff, Vb + goff, in ? 16 : 0);
            }
            sm90_cp_commit();
        };
        /* Consume the landing signal for buffer `buf`: mbarrier phase if TMA staged it,
         * cp.async group drain otherwise (Q rides the cp path in both cases). */
        auto waitKV = [&](int buf) {
            sm90_cp_wait<0>();
            if (fa90_tma[buf]) {
                sm90_mbar_wait(&fa90_bar[buf], (int)(fa90_ph[buf] & 1u));
                fa90_ph[buf]++;
                fa90_tma[buf] = false; /* consumed — a skipped restage must not re-wait */
            }
        };

        __syncthreads(); /* previous item's Qs/Ks/Vs/Ps reads complete before restaging */

        for (int i = tid; i < BQ * NSUB * 8; i += (int)PLOW_NV_THREADS) {
            const int c = i & 7, sub = (i >> 3) % NSUB, r = i / (8 * NSUB);
            const bool in = (q0 + (unsigned)r < sq);
            sm90_cp16(Qs + sub * QT + sm90_swz_off<64, 8>(r, c),
                      Qh + (in ? (size_t)r * n_head * HD + (size_t)sub * 64 + (size_t)c * 8 : 0),
                      in ? 16 : 0);
        }
        sm90_cp_commit();
        if (ntile > 0) stageKV(eff_lo, 0);
        waitKV(0);

        /* Online state, REGISTERS ONLY (finding (a)): this lane's two rows, replicated across its
         * quad and kept bit-identical by the butterflies below. */
        float mA = FA_NEG_INF, lA = 0.0f, mB = FA_NEG_INF, lB = 0.0f;
        float Oacc[NTW][32];
#pragma unroll
        for (int t = 0; t < NTW; t++)
#pragma unroll
            for (int i = 0; i < 32; i++) Oacc[t][i] = 0.0f;

        const int qabsA = (int)(qp0 + q0) + rA, qabsB = (int)(qp0 + q0) + rB;
        int sb = 0;

        for (int t = 0; t < ntile; t++) {
            const unsigned kv0 = eff_lo + (unsigned)t * BKV;
            __syncthreads(); /* (A) tile t visible block-wide; tile t-1's buffer + Ps now free */
            fa90_async_proxy_fence();
            if (t + 1 < ntile) stageKV(kv0 + BKV, sb ^ 1);

            /* --- GEMM0: S[64][BKV] = Q . K^T. scale-d = 0 on the first k-step seeds the
             * accumulator, so no zeroing pass. --- */
            const __nv_bfloat16* kbuf = Ks + (size_t)sb * NSUB * KT;
            float S[BKV / 2];
            sm90_wg_fence();
#pragma unroll 1
            for (int ks = 0; ks < KS0; ks++) {
                const int sub = ks >> 2, ko = (ks & 3) * 16; /* +32 B per k16 substep */
                fa90_wgmma_score<BKV>(S, sm90_desc(Qs + sub * QT + ko),
                                      sm90_desc(kbuf + sub * KT + ko), ks ? 1 : 0);
            }
            sm90_wg_commit();
            sm90_wg_wait<0>();

            /* --- scale + causal/sliding mask IN THE FRAGMENT --- */
            float mxA = FA_NEG_INF, mxB = FA_NEG_INF;
#pragma unroll
            for (int nb = 0; nb < NB0; nb++)
#pragma unroll
                for (int e = 0; e < 2; e++) {
                    const int kv = (int)kv0 + 8 * nb + 2 * (lane & 3) + e;
                    const bool inr = ((unsigned)kv < hi);
                    bool okA = inr && (kv <= qabsA), okB = inr && (kv <= qabsB);
                    if (window) {
                        okA = okA && ((unsigned)(qabsA - kv) < window);
                        okB = okB && ((unsigned)(qabsB - kv) < window);
                    }
                    const float a = okA ? S[4 * nb + e] * lscale : FA_NEG_INF;
                    const float b = okB ? S[4 * nb + 2 + e] * lscale : FA_NEG_INF;
                    S[4 * nb + e] = a;
                    S[4 * nb + 2 + e] = b;
                    mxA = fmaxf(mxA, a);
                    mxB = fmaxf(mxB, b);
                }
            /* QUAD butterfly: a row's BKV columns are split over the 4 lanes sharing (lane>>2),
             * so xor-1 then xor-2 completes the row reduction and leaves all 4 copies equal. */
            mxA = fmaxf(mxA, __shfl_xor_sync(0xffffffffu, mxA, 1));
            mxA = fmaxf(mxA, __shfl_xor_sync(0xffffffffu, mxA, 2));
            mxB = fmaxf(mxB, __shfl_xor_sync(0xffffffffu, mxB, 1));
            mxB = fmaxf(mxB, __shfl_xor_sync(0xffffffffu, mxB, 2));

            const float mnA = fmaxf(mA, mxA), mnB = fmaxf(mB, mxB);
            /* corr==0 on the first attended tile (m_old still -inf) — the shared body's rule. */
            const float cA = (mA == FA_NEG_INF) ? 0.0f : FA_EXP(mA - mnA);
            const float cB = (mB == FA_NEG_INF) ? 0.0f : FA_EXP(mB - mnB);
            mA = mnA;
            mB = mnB;

            /* P -> the GEMM1 A operand. The two e-halves are adjacent columns of the same core
             * matrix, so each pair is one 32-bit bf16x2 store. Both warpgroups compute identical
             * P; only wg 0 stores it. */
            const bool liveA = (mnA != FA_NEG_INF), liveB = (mnB != FA_NEG_INF);
            float sA = 0.0f, sB = 0.0f;
#pragma unroll
            for (int nb = 0; nb < NB0; nb++) {
                const float pa0 = liveA ? FA_EXP(S[4 * nb + 0] - mnA) : 0.0f;
                const float pa1 = liveA ? FA_EXP(S[4 * nb + 1] - mnA) : 0.0f;
                const float pb0 = liveB ? FA_EXP(S[4 * nb + 2] - mnB) : 0.0f;
                const float pb1 = liveB ? FA_EXP(S[4 * nb + 3] - mnB) : 0.0f;
                sA += pa0 + pa1;
                sB += pb0 + pb1;
                if (wg == 0) {
                    const int c0 = 8 * nb + 2 * (lane & 3);
                    *(__nv_bfloat162*)(Ps + fa90_cm_off<BKV>(rA, c0)) =
                        __floats2bfloat162_rn(pa0, pa1);
                    *(__nv_bfloat162*)(Ps + fa90_cm_off<BKV>(rB, c0)) =
                        __floats2bfloat162_rn(pb0, pb1);
                }
            }
            sA += __shfl_xor_sync(0xffffffffu, sA, 1);
            sA += __shfl_xor_sync(0xffffffffu, sA, 2);
            sB += __shfl_xor_sync(0xffffffffu, sB, 1);
            sB += __shfl_xor_sync(0xffffffffu, sB, 2);
            lA = lA * cA + sA;
            lB = lB * cB + sB;

            /* Rescale O on the SAME two rows this lane owns — two scalars, no smem, and by
             * construction the same (row,col) map GEMM0 used. */
#pragma unroll
            for (int nt = 0; nt < NTW; nt++)
#pragma unroll
                for (int nb = 0; nb < 8; nb++)
#pragma unroll
                    for (int e = 0; e < 2; e++) {
                        Oacc[nt][4 * nb + e] *= cA;
                        Oacc[nt][4 * nb + 2 + e] *= cB;
                    }

            /* Ps is written above with GENERIC stores by wg0; GEMM1 reads it with an ASYNC-PROXY
             * wgmma from BOTH warpgroups. Order is intentional and probe-validated: barrier (B)
             * first publishes wg0's generic stores to the whole block, then each consumer's OWN
             * fa90_async_proxy_fence orders those now-visible generic writes ahead of ITS following
             * async-proxy wgmma. This is the per-consumer fence pattern, sound under PTX fence
             * cumulativity — NOT the CUTLASS producer-side order (fence-before-barrier), which would
             * only fence wg0. Do not swap these two lines: moving the fence above the barrier drops
             * the ordering guarantee for wg1's reads of wg0's Ps. */
            __syncthreads(); /* (B) Ps published to the whole block */
            fa90_async_proxy_fence();

            /* --- GEMM1: O += P . V. A = Ps (K-major over BKV, swizzle-0); B = V MN-MAJOR
             * (trans-b=1) in its natural [BKV][HD] layout, sub-tile g IS the n64 output tile,
             * k-slice ks = rows [16ks, 16ks+16) = +ks*16*64 elements. --- */
            const __nv_bfloat16* vbuf = Vs + (size_t)sb * NSUB * KT;
            sm90_wg_fence();
#pragma unroll
            for (int nt = 0; nt < NTW; nt++) {
                const int g = wg * NTW + nt;
#pragma unroll 1
                for (int ks = 0; ks < KS1; ks++)
                    fa90_wgmma_m64n64k16_tb1(Oacc[nt],
                                             fa90_desc_ns(Ps + ks * 128, 128, 16 * BKV),
                                             sm90_desc(vbuf + g * KT + ks * (16 * 64)), 1);
            }
            sm90_wg_commit();
            sm90_wg_wait<0>();
            waitKV(sb ^ 1); /* tile t+1 has landed for this thread */
            sb ^= 1;
        }

        /* --- epilogue. nsplit>1: UNNORMALISED partials + (m,l) for d_flash_merge. Otherwise
         * normalise by l and write bf16. Same (row,col) map, so no re-derivation. --- */
        const float iA = (lA > 0.0f) ? 1.0f / lA : 0.0f;
        const float iB = (lB > 0.0f) ? 1.0f / lB : 0.0f;
#pragma unroll
        for (int nt = 0; nt < NTW; nt++) {
            const int g = wg * NTW + nt;
#pragma unroll
            for (int nb = 0; nb < 8; nb++)
#pragma unroll
                for (int e = 0; e < 2; e++) {
                    const int hd = g * 64 + 8 * nb + 2 * (lane & 3) + e;
                    const unsigned raA = q0 + (unsigned)rA, raB = q0 + (unsigned)rB;
                    const float va = Oacc[nt][4 * nb + e], vb = Oacc[nt][4 * nb + 2 + e];
                    if (ns > 1) {
                        if (raA < sq)
                            Opart[((size_t)(raA * n_head + h) * ns + sp) * HD + hd] = va;
                        if (raB < sq)
                            Opart[((size_t)(raB * n_head + h) * ns + sp) * HD + hd] = vb;
                    } else {
                        if (raA < sq)
                            O[qoff + (size_t)(raA * n_head + h) * HD + hd] =
                                __float2bfloat16(va * iA);
                        if (raB < sq)
                            O[qoff + (size_t)(raB * n_head + h) * HD + hd] =
                                __float2bfloat16(vb * iB);
                    }
                }
        }
        /* (m,l) partials: rows rA/rB are replicated across the quad — lane%4==0 of warpgroup 0
         * writes each row exactly once, and (w, lane>>2) x {rA,rB} covers all BQ rows. */
        if (ns > 1 && wg == 0 && (lane & 3) == 0) {
            const unsigned raA = q0 + (unsigned)rA, raB = q0 + (unsigned)rB;
            if (raA < sq) {
                float* ml = mlpart + ((size_t)(raA * n_head + h) * ns + sp) * 2;
                ml[0] = mA;
                ml[1] = lA;
            }
            if (raB < sq) {
                float* ml = mlpart + ((size_t)(raB * n_head + h) * ns + sp) * 2;
                ml[0] = mB;
                ml[1] = lB;
            }
        }
    }
}

#endif /* PLOW_OP_ATTENTION_SM90_CUH */
