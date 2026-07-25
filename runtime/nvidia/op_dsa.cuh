/* op_dsa.cuh — DeepSeek Sparse Attention (DSA) top-k SELECT + gathered attention for sm_120a.
 *
 * Port of the MI350X/gfx950 DSA path (runtime/amd/op_attention.h: d_index_select_coop,
 * dsa_pack_key_a, dsa_grid_sync_t, d_flash_mla_decode<...,GATHER=true>) onto RTX 5090 / GB202.
 *
 * WHAT DSA IS (see runtime/amd/dsa-gather-design.md for the full derivation):
 *   1. INDEXER   score[t] = sum_h w[h] * ReLU( q_idx[h] . k_idx[t] )     (DeepSeek-V3.2 eq.1)
 *      a [HI=32 x DI=128] . [DI x ctx] GEMM + w-weighted ReLU head reduction. Not in this file.
 *   2. SELECT    top_k=2048 positions of score[], EXACT, lowest-index tie-break  -> idx[top_k].
 *   3. GATHER    attention restricted to those top_k KV rows: cost is CONSTANT in ctx, whereas
 *      dense attention is ctx-linear. That constancy is the entire lever.
 *
 * PORTING NOTES (AMD -> sm_120), each one a place a transliteration would have been wrong:
 *
 * [W32] Warp is 32 lanes, not 64. The AMD select kernel happens to contain NO wave reduction
 *       (its cross-thread communication is all LDS/global atomics + threadIdx.x==0 scalars), so it
 *       ports structurally. The AMD *indexer* epilogue does `__shfl_xor(part, 32, 64)` to fold two
 *       wave64 head-halves — that has no 32-lane analogue and must be re-derived from the mma.sync
 *       fragment layout, NOT transliterated. The gather kernel here uses a 5-step (log2 32)
 *       __shfl_xor_sync butterfly, re-derived for 32 lanes.
 *
 * [COH] The AMD barrier deliberately omits any fence and communicates the histogram purely through
 *       L2-coherent atomics. That reasoning TRANSFERS to NVIDIA only if every cross-block read
 *       bypasses the (non-coherent) L1: atomicAdd(&x,0u) does, a plain load does not. So the
 *       ATOMIC_SYNC + PAR_SCAN configuration — which AMD shipped for *performance* — is on sm_120
 *       also the *correctness* configuration. Every cross-block read below is an atomic.
 *
 * [SCAN] Commit 70ddc9c's lesson, which applies verbatim here: the AMD select was assumed to be
 *       grid-barrier bound and was NOT. The real cost was a serial lane-0 chain of up to 256
 *       dependent global reads (the per-pass boundary digit scan) — a latency chain, run redundantly
 *       by every block. The sm_120 trap is identical in shape: blaming __syncthreads()/grid sync when
 *       the cost is a serial dependent scan. PAR_SCAN keeps 256 reads in flight instead of 256 in a
 *       chain. Profile (ncu) before touching the barrier.
 *
 * [NWG] The AMD "32-WG select" is a tuning constant for a 256-CU MI350X (1/8 of the chip). This box
 *       has 170 SMs and a different atomic/L2 subsystem; the constant is NOT transferable and is
 *       re-derived by measurement (DSA_SELWG sweep in test_dsa.cu). It is a runtime argument here,
 *       never a hardcoded 32.
 *
 * [MFMA] The AMD indexer (d_index_score_mfma) uses MFMA — an AMD matrix core with NO NVIDIA
 *       equivalent. What it computes is a [HI=32 x DI=128] . [DI x ctx] GEMM contracted over DI,
 *       then a w-weighted ReLU reduction over the 32 heads. The sm_120 replacement is derived from
 *       the roofline, NOT by defaulting to tensor cores:
 *
 *         work  @ctx=128k = 2*HI*DI*ctx      = 1.074 GFLOP
 *         bytes @ctx=128k = ctx*DI*2 (bf16)  = 33.55 MB   (Q is 8 KiB, negligible)
 *         arithmetic intensity               = 32.0 flop/byte
 *         machine balance, FFMA  = 104.9 TF/s / 1.79 TB/s = 58.6 flop/byte
 *         machine balance, mma   = 209.5 TF/s / 1.79 TB/s = 117  flop/byte
 *
 *       32.0 < 58.6 < 117, so the kernel is HBM-BOUND under EITHER math unit — the HBM floor is
 *       33.55 MB / 1.79 TB/s = 18.7 us, against 10.2 us of FFMA or 5.1 us of mma.sync. FLOPs alone
 *       therefore do NOT decide it, and "use tensor cores" is not automatically right.
 *
 *       What decides it is issue pressure against the load stream. The tile is NOT too small for
 *       tensor cores: M=HI=32 is two m16 tiles, K=DI=128 is eight full k16 steps, N=positions is
 *       free. Per warp per 8 positions, mma.sync m16n8k16 does 2*8 = 16 instructions for what FFMA
 *       needs 32768/32 = 1024 instructions to do — 64x fewer issue slots, which is the headroom
 *       required to keep the coalesced K stream in flight. So: mma.sync m16n8k16, but for issue
 *       bandwidth, not throughput.
 *
 *       The REAL lever is the same one AMD found, and is orthogonal to the math unit: one thread
 *       per position makes each lane read its own 256 B key, so every 128 B sector serves a single
 *       lane and effective bandwidth collapses (AMD measured ~350 GB/s of ~5 TB/s). K must be
 *       streamed contiguously through shared memory first; only then does the choice of math unit
 *       matter at all.
 *
 *       [W32] The AMD epilogue's `__shfl_xor(part, 32, 64)` exists because MFMA 32x32x16 leaves
 *       complementary head-halves in lanes l and l+32 of a wave64. On sm_120 the m16n8k16 fragment
 *       layout is different and the two m16 head-halves are SEPARATE mma instructions with separate
 *       accumulators — so the fold is a plain register add, with NO shuffle. Transliterating the
 *       shfl_xor here would be silently wrong.
 *
 * [RAG] Ragged/short selection sets. The AMD kernel assumes len > top_k (guaranteed upstream by the
 *       ctx > 65536 emit gate with top_k=2048). If len <= top_k it never satisfies k_rem, leaves
 *       idx[len..top_k) UNWRITTEN, and the gather then reads uninitialised indices out of bounds.
 *       This port takes an explicit early-out and writes an authoritative n_sel count that the
 *       gather honours, so a ragged set is handled rather than assumed away.
 */
#ifndef PLOW_OP_DSA_CUH
#define PLOW_OP_DSA_CUH

#include <cuda_bf16.h>
#include <stdint.h>

#ifndef DSA_THREADS
#define DSA_THREADS 256u /* threads/block for the selector (8 warps of 32) */
#endif
#ifndef PLOW_THREADS
#define PLOW_THREADS 256 /* block width for d_index_score_sm120 (8 warps of 32) */
#endif

typedef __nv_bfloat16 dsa_bf16;
__device__ __forceinline__ float dsa_b2f(dsa_bf16 v) { return __bfloat162float(v); }
struct __align__(16) dsa_bf16v8 { dsa_bf16 v[8]; };

/* ---------------------------------------------------------------------------------------------
 * INDEXER SCORE (V4, PLOW_DOP_INDEX_SCORE=58) — sm_120 port of the AMD d_index_score_mfma.
 *
 *   score[t] = scale * Σ_h w[h] * ReLU( q_idx[h] . k_idx[t] )      HI=32 heads, DI=128 dim.
 *
 * WHY mma.sync (per the [MFMA] roofline note above): the score-dot is HBM-BOUND under either math
 * unit, but one thread per position reads its own DI-strided 256 B key so effective BW collapses.
 * K is therefore STREAMED CONTIGUOUSLY through smem, and the [HI x pos] dot falls out of tensor
 * cores at 64x fewer issue slots than scalar FFMA — issue bandwidth, not throughput, is the lever.
 *
 * TILING. mma.sync.m16n8k16 bf16 (M=head, K=DI, N=pos). HI=32 => MT=2 m16 tiles; DI=128 => 8 k16
 * steps; positions tiled 8 per mma. A=Qidx[head][DI] row-major (ldmatrix.x4). B=Kidx[pos][DI]
 * staged NATURAL and read with ldmatrix.x2 NON-.trans — the T5 flash-prefill K-load layout
 * (op_attention.cuh:1198), which is bit-identical to the transposed K^T operand. The D accumulator
 * (head m, pos n) is written to smem in [head][pos] layout using the EXACT (qr,kc) fragment map the
 * prefill QK^T validates (op_attention.cuh:1205-1206), then a plain per-position loop does the
 * w-weighted ReLU head reduction from smem — no fragile cross-lane fold. mma is used only for the
 * q.k dot (the issue-critical part); the head reduction is a cheap smem pass. HMMA present in SASS.
 * ------------------------------------------------------------------------------------------- */
#define DSA_SCORE_TILE_N 64 /* KV positions per slab (NWARPS * positions-per-warp) */
/* smem floats for d_index_score_sm120<DI,HI>: wlds[HI] + qlds[HI][DI+8]bf16 + ktile[TN][DI+8]bf16 +
 * Dsm[HI][TN]f32. TN=64 -> 8608 f = 33.6 KiB. */
#define DSA_SCORE_SMEM_FLOATS(DI, HI, TN)                                                          \
    ((HI) + (HI) * ((DI) + 8) / 2 + (TN) * ((DI) + 8) / 2 + (HI) * (TN))

__device__ __forceinline__ void dsa_ldmatrix_x4(unsigned (&r)[4], const void* smem) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
__device__ __forceinline__ void dsa_ldmatrix_x2(unsigned (&r)[2], const void* smem) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r[0]), "=r"(r[1]) : "r"(s));
}
__device__ __forceinline__ void dsa_mma(float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2],
                                        const float (&c)[4]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]),
                   "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

/* Operand contract (emitter gemma4.rs:4145 / dev.rs:417 / dev_isa.h:400):
 *   t0=Score(f32 [b][kv_stride])  t1=Qidx(bf16 [b][HI][DI])  t2=Kidx(bf16 [b][kv_stride][DI])
 *   t3=W(bf16 [b][HI])            t4=kv_len(i32[b]);   i0=n_batch i2=kv_stride ; f0=scale
 * HI/DI are model constants (GLM index_n_heads=32 / index_head_dim=128), template args. */
template <int DI, int HI>
__device__ void d_index_score_sm120(float* __restrict__ Score, const dsa_bf16* __restrict__ Qidx,
                                    const dsa_bf16* __restrict__ Kidx, const dsa_bf16* __restrict__ W,
                                    const int* __restrict__ kv_len, unsigned n_batch,
                                    unsigned kv_stride, float scale, unsigned slice, unsigned nblk,
                                    float* smem) {
    static_assert(DI % 16 == 0, "DI must be a whole number of mma k16 steps");
    static_assert(HI % 16 == 0, "HI must be a whole number of m16 tiles");
    constexpr int PAD = 8;                        /* smem row pad (bank-conflict break) */
    constexpr int KSTEPS = DI / 16;               /* wide-K contraction steps (DI=128 -> 8) */
    constexpr int MT = HI / 16;                   /* m16 head tiles (HI=32 -> 2) */
    constexpr int NWARPS = PLOW_THREADS / 32;     /* 8 */
    constexpr int TILE_N = DSA_SCORE_TILE_N;      /* 64 */
    constexpr int NPT = TILE_N / NWARPS;          /* positions per warp (8) */
    constexpr int NN = NPT / 8;                   /* n8 mma groups per warp (1) */
    static_assert(NPT % 8 == 0, "positions per warp must be a whole number of n8 mma tiles");

    float* wlds = smem;                                       /* [HI] head weights */
    dsa_bf16* qlds = (dsa_bf16*)(wlds + HI);                  /* [HI][DI+PAD] absorbed query rows */
    dsa_bf16* ktile = qlds + HI * (DI + PAD);                 /* [TILE_N][DI+PAD] streamed key slab */
    float* Dsm = (float*)(ktile + TILE_N * (DI + PAD));       /* [HI][TILE_N] score-dot dump */

    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 31u, warp = tid >> 5;

    for (unsigned b = 0; b < n_batch; b++) {
        const unsigned len = (unsigned)kv_len[b];
        __syncthreads();
        /* stage the HI query rows (padded) + the HI head weights; reused across every key slab. */
        for (unsigned e = tid; e < (unsigned)HI * (DI / 8); e += PLOW_THREADS) {
            const unsigned h = e / (DI / 8), c8 = (e % (DI / 8)) * 8;
            *(dsa_bf16v8*)&qlds[h * (DI + PAD) + c8] =
                *(const dsa_bf16v8*)&Qidx[(size_t)b * HI * DI + h * DI + c8];
        }
        for (unsigned i = tid; i < (unsigned)HI; i += PLOW_THREADS)
            wlds[i] = dsa_b2f(W[(size_t)b * HI + i]);
        __syncthreads();

        const unsigned nslab = (len + TILE_N - 1) / TILE_N;
        for (unsigned st = slice; st < nslab; st += nblk) {
            const unsigned base = st * TILE_N;
            __syncthreads(); /* previous slab's ktile/Dsm readers done before overwrite */
            /* coalesced contiguous key load [TILE_N][DI], zero-filled past len. */
            for (unsigned c = tid; c < (unsigned)TILE_N * (DI / 8); c += PLOW_THREADS) {
                const unsigned row = c / (DI / 8), c8 = (c % (DI / 8)) * 8;
                const unsigned pos = base + row;
                dsa_bf16v8 kv;
                if (pos < len) {
                    kv = *(const dsa_bf16v8*)&Kidx[((size_t)b * kv_stride + pos) * DI + c8];
                } else {
#pragma unroll
                    for (int j = 0; j < 8; j++) kv.v[j] = __float2bfloat16(0.0f);
                }
                *(dsa_bf16v8*)&ktile[row * (DI + PAD) + c8] = kv;
            }
            __syncthreads(); /* key slab visible to all warps before the mma reads */

            /* this warp's NPT positions: D[head][pos] = Qidx . Kidx over DI, into Dsm. */
#pragma unroll
            for (int mt = 0; mt < MT; mt++) {
#pragma unroll
                for (int ng = 0; ng < NN; ng++) {
                    const unsigned n0 = warp * NPT + (unsigned)ng * 8;
                    float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                    for (int kf = 0; kf < KSTEPS; kf++) {
                        unsigned af[4];
                        dsa_ldmatrix_x4(af, &qlds[(mt * 16 + (lane % 16)) * (DI + PAD) + kf * 16 +
                                                  (lane / 16) * 8]);
                        unsigned bf[2];
                        const unsigned n = n0 + (lane & 7u);
                        const unsigned kcol = kf * 16 + ((lane >> 3) & 1u) * 8;
                        dsa_ldmatrix_x2(bf, &ktile[n * (DI + PAD) + kcol]);
                        dsa_mma(acc, af, bf, acc);
                    }
#pragma unroll
                    for (int e = 0; e < 4; e++) {
                        const unsigned h = mt * 16 + (lane / 4) + (unsigned)(e / 2) * 8;
                        const unsigned nn = n0 + (lane % 4) * 2 + (unsigned)(e % 2);
                        Dsm[h * TILE_N + nn] = acc[e];
                    }
                }
            }
            __syncthreads(); /* full [head][pos] tile visible before the head reduction */

            /* w-weighted ReLU head reduction: thread `tid` owns strided positions of this slab. */
            const unsigned rmax = (len - base < (unsigned)TILE_N) ? (len - base) : (unsigned)TILE_N;
            for (unsigned p = tid; p < rmax; p += PLOW_THREADS) {
                float s = 0.0f;
#pragma unroll
                for (int h = 0; h < HI; h++) {
                    const float d = Dsm[h * TILE_N + p];
                    s += wlds[h] * (d > 0.0f ? d : 0.0f);
                }
                Score[(size_t)b * kv_stride + base + p] = s * scale;
            }
        }
    }
}

/* ---------------------------------------------------------------------------------------------
 * Packed key. Monotone map of (score desc, index asc) into one u64, so "the top_k largest keys"
 * == "the top_k highest scores, ties broken by LOWEST index". The low 24 index bits make every key
 * UNIQUE, which is what lets a threshold select land on exactly top_k with no tie bookkeeping.
 *
 * Byte-ALIGNED (dsa_pack_key_a): the 32 score bits occupy exactly radix bytes 0..3 and the index
 * bytes 4..6. So after 4 passes the score threshold is fully resolved, and absent a genuine
 * exact-score tie straddling the boundary the selection is decided in 4 passes, not 7.
 * ordered_bits: flip the sign bit for positives, invert everything for negatives — the standard
 * IEEE-754 -> unsigned order-preserving transform (handles -0.0/negatives correctly).
 * ------------------------------------------------------------------------------------------- */
__device__ __forceinline__ unsigned long long dsa_pack_key_a(float sc, unsigned t, unsigned len) {
    unsigned sb = __float_as_uint(sc);
    sb = (sb & 0x80000000u) ? ~sb : (sb | 0x80000000u);
    return ((unsigned long long)sb << 24) | (unsigned long long)((len - 1u - t) & 0xFFFFFFu);
}

/* ---------------------------------------------------------------------------------------------
 * Sense-reversing grid barrier over CO-RESIDENT blocks (requires a cooperative launch, or
 * gridDim <= the occupancy-derived resident capacity). Only thread 0 of each block participates.
 *
 * [COH] No __threadfence(). The histogram is communicated EXCLUSIVELY through atomics, which are
 * performed at L2 (device-coherent) and bypass the non-coherent L1. This block's histogram
 * atomicAdds precede its arrival atomicAdd in program order, and every reader reads the histogram
 * with an atomic too — so the arrival handshake alone orders writers-before-readers. Polling `gen`
 * with atomicAdd(&,0u) (rather than a plain/volatile load) is what makes the release visible at L2
 * latency instead of write-back latency, AND is what keeps a stale L1 line from being observed.
 * ------------------------------------------------------------------------------------------- */
__device__ __forceinline__ void dsa_grid_sync(unsigned* ctl, unsigned nwg) {
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned g = atomicAdd(&ctl[1], 0u); /* coherent read of the generation */
        const unsigned a = atomicAdd(&ctl[0], 1u) + 1u;
        if (a == nwg) {                            /* last arriver: reset count, bump generation */
            atomicExch(&ctl[0], 0u);
            atomicExch(&ctl[1], g + 1u);
        } else {
            while (atomicAdd(&ctl[1], 0u) == g) { __nanosleep(64); }
        }
    }
    __syncthreads();
}

/* Radix geometry. 8-bit / 256-bin / 7-pass, matching AMD — wider digits LOST there (the bigger
 * histogram's flush + read-back traffic dwarfs the barriers saved), and that tradeoff is
 * architecture-independent enough to inherit rather than re-sweep. */
#define DSA_SEL_DIGIT 8u
#define DSA_SEL_NB    (1u << DSA_SEL_DIGIT) /* 256 bins per pass                       */
#define DSA_SEL_NPASS 7u                    /* 4 score bytes + 3 index bytes = 56 bits */

/* ---------------------------------------------------------------------------------------------
 * DSA top-k SELECT — MSB-first 8-bit radix threshold select, ONE cooperative launch.
 *
 * Per pass every block histograms its grid-strided slice of the surviving keys into a 256-bin
 * shared-memory table, flushes to the global histogram, grid-syncs, then ALL blocks independently
 * read the completed histogram and compute the SAME boundary digit (no single decider => the
 * running prefix/k_rem stay in registers, no per-pass HBM state). After the last pass `prefix` is
 * the top_k-th largest key and every position with key >= prefix is emitted.
 *
 * Emission ORDER is immaterial: attention's online softmax is set-order invariant.
 *
 *   idx    [top_k]  out, selected positions
 *   n_sel  [1]      out, how many are actually valid ([RAG]: min(top_k, len))
 *   Score  [len]    in,  f32 indexer scores
 *   gHist  [NPASS*NB] scratch, zeroed by the kernel itself (idempotent)
 *   gCtl   [3]      scratch, {arrive, generation, emit-slot}; host zeroes ONCE before first launch
 *   nwg    == gridDim.x, must be co-resident
 * ------------------------------------------------------------------------------------------- */
/* INTERP-WRAP (V5, PLOW_DOP_INDEX_SELECT=59): a __device__ body (was a standalone __global__) so
 * the megakernel switch can inline it. The block id `bid` and co-resident count `nwg` are the
 * PACKET's (slice, blocks) — the emitter emits this on exactly `blocks` CUs (gemma4.rs:4162, 32),
 * so slice ranges 0..blocks-1 and nwg==blocks, exactly the grid the fenceless L2-atomic barrier
 * (dsa_grid_sync) needs. Under a standalone cooperative launch (the device oracle) bid=blockIdx.x,
 * nwg=gridDim.x. `n_sel` is OPTIONAL: the emitter provides no n_sel tensor (the gather reads top_k
 * directly and ctx>65536 guarantees len>top_k), so it is nullptr-guarded. */
template <bool PAR_SCAN = true>
__device__ void d_index_select_sm120(
    int* __restrict__ idx, unsigned* __restrict__ n_sel, const float* __restrict__ Score,
    unsigned len, unsigned top_k, unsigned* __restrict__ gHist, unsigned* __restrict__ gCtl,
    unsigned bid, unsigned nwg) {
    __shared__ unsigned lh[DSA_SEL_NB];
    __shared__ unsigned red[3];
    const unsigned tid = threadIdx.x;

    /* [RAG] Ragged / short context: fewer positions than requested. Select ALL of them (identity)
     * and report the true count. The radix path below cannot satisfy k_rem = top_k > len and would
     * silently leave idx[len..top_k) uninitialised — the out-of-bounds gather the AMD kernel is
     * saved from only by its upstream ctx > 65536 gate. */
    if (len <= top_k) {
        if (bid == 0 && tid == 0 && n_sel) *n_sel = len;
        for (unsigned t = bid * DSA_THREADS + tid; t < len; t += nwg * DSA_THREADS) idx[t] = (int)t;
        return;
    }
    if (bid == 0 && tid == 0) {
        atomicExch(&gCtl[2], 0u); /* reset emit slot */
        if (n_sel) *n_sel = top_k;
    }
    /* clear every pass's histogram cooperatively (idempotent across blocks), then sync. */
    for (unsigned i = bid * DSA_THREADS + tid; i < DSA_SEL_NPASS * DSA_SEL_NB; i += nwg * DSA_THREADS)
        atomicExch(&gHist[i], 0u);
    dsa_grid_sync(gCtl, nwg);

    unsigned long long prefix = 0ull, himask = 0ull;
    unsigned k_rem = top_k;
    for (unsigned pass = 0; pass < DSA_SEL_NPASS; pass++) {
        const unsigned sh = (DSA_SEL_NPASS - 1u - pass) * DSA_SEL_DIGIT;
        unsigned* const Hp = gHist + (size_t)pass * DSA_SEL_NB;

        for (unsigned i = tid; i < DSA_SEL_NB; i += DSA_THREADS) lh[i] = 0u;
        __syncthreads();
        /* histogram this block's slice of the keys still matching the resolved prefix. */
        for (unsigned t = bid * DSA_THREADS + tid; t < len; t += nwg * DSA_THREADS) {
            const unsigned long long key = dsa_pack_key_a(Score[t], t, len);
            if ((key & himask) == prefix)
                atomicAdd(&lh[(unsigned)((key >> sh) & (DSA_SEL_NB - 1u))], 1u);
        }
        __syncthreads();
        for (unsigned i = tid; i < DSA_SEL_NB; i += DSA_THREADS)
            if (lh[i]) atomicAdd(&Hp[i], lh[i]);
        dsa_grid_sync(gCtl, nwg); /* every block has contributed to Hp */

        /* [SCAN] Boundary digit: highest digit d such that the top-down cumulative count crosses
         * k_rem. PARALLEL coherent read-back of all 256 bins into shared memory (256 loads in
         * flight), THEN a serial accumulate over shared memory. The AMD baseline instead ran a
         * serial lane-0 chain of 256 dependent global atomic reads and that chain — not the
         * barrier — was the measured bottleneck (commit 70ddc9c, 2.1-3.4x). */
        if (PAR_SCAN) {
            for (unsigned i = tid; i < DSA_SEL_NB; i += DSA_THREADS) lh[i] = atomicAdd(&Hp[i], 0u);
            __syncthreads();
            if (tid == 0) {
                unsigned acc = 0u, dsel = 0u, bnd = 0u;
                for (int d = (int)DSA_SEL_NB - 1; d >= 0; d--) {
                    const unsigned hd = lh[d];
                    if (acc + hd >= k_rem) { dsel = (unsigned)d; bnd = hd; break; }
                    acc += hd;
                }
                red[0] = dsel; red[1] = acc; red[2] = bnd;
            }
        } else {
            /* BASELINE (the AMD pre-70ddc9c form): a SERIAL thread-0 chain of up to 256 dependent
             * global atomic reads, each a full L2 round-trip, run redundantly by every block.
             * Kept instantiated so the sm_120 measurement can confirm or refute that this — and not
             * the grid barrier — is the cost here too. */
            if (tid == 0) {
                unsigned acc = 0u, dsel = 0u, bnd = 0u;
                for (int d = (int)DSA_SEL_NB - 1; d >= 0; d--) {
                    const unsigned hd = atomicAdd(&Hp[d], 0u);
                    if (acc + hd >= k_rem) { dsel = (unsigned)d; bnd = hd; break; }
                    acc += hd;
                }
                red[0] = dsel; red[1] = acc; red[2] = bnd;
            }
        }
        __syncthreads();
        k_rem -= red[1];
        prefix |= ((unsigned long long)red[0] << sh);
        himask |= ((unsigned long long)(DSA_SEL_NB - 1u) << sh);

        /* Fewer-passes early-out: after the 4 SCORE bytes the 32-bit score threshold is fully
         * resolved. k_rem is now how many are still needed from the boundary bin, whose population
         * is red[2] — all sharing the EXACT boundary score. If they match, the whole tied group is
         * taken and the 3 index passes would emit the same set. Only a genuine score tie that must
         * be SPLIT by index (red[2] > k_rem) continues. Every block reads the same red[] so the
         * branch is grid-uniform and no block is left waiting at a barrier the others skipped. */
        if (pass == 3u && red[2] == k_rem) break;
    }
    /* Unique keys => #{key >= prefix} == top_k exactly. */
    for (unsigned t = bid * DSA_THREADS + tid; t < len; t += nwg * DSA_THREADS) {
        if (dsa_pack_key_a(Score[t], t, len) >= prefix) {
            const unsigned slot = atomicAdd(&gCtl[2], 1u);
            if (slot < top_k) idx[slot] = (int)t;
        }
    }
}

/* ---------------------------------------------------------------------------------------------
 * NOT WIRED — reconcile note (P3): d_gather_attn_decode / d_gather_merge below are the DENSE-K/V
 * gather (per-head K/V reconstruction), the WRONG gather for the absorbed MLA op-record. The
 * absorbed DSA gather is op_mla.cuh d_flash_mla_decode_sm120<...,GATHER=true> (V2, verified in P1).
 * The interp dispatch does NOT reference these two; only d_index_select_sm120 (above) and
 * d_index_score_sm120 are used from this file. Kept for reference / a future explicit-form path.
 *
 * GATHERED flash-attention decode (one query token) over the selected rows.
 *
 *   o[h][d] = sum_r softmax_r( scale * q[h] . K[idx[r]] ) * V[idx[r]][d]
 *
 * K/V are a SHARED (MQA/MLA-latent style) cache [ctx][D]; heads differ only in q. This is why the
 * gather is constant-cost: it always touches n_sel*D*2 bytes regardless of ctx.
 *
 * Split-K over the selected set so the grid fills 170 SMs: grid = (nsplit, n_head). Each block
 * online-softmaxes its own contiguous slice of idx[] and writes (partial acc, m, l); d_gather_merge
 * combines them. Online softmax is associative, so any split is exact.
 *
 * Block = D threads (D=128 -> 4 warps of 32).
 *   Phase A: score TN rows. Warp w handles rows w, w+NW, ... ; each of the 32 lanes reduces D/32
 *            elements and a 5-step (log2 32 — [W32], re-derived, NOT the wave64 6-step) __shfl_xor
 *            butterfly folds the lane partials.
 *   Phase B: thread d owns output dim d and walks the TN rows; V[idx[r]][d] across the block is
 *            contiguous in d, so each row's V read is a fully coalesced 256 B line.
 * ------------------------------------------------------------------------------------------- */
#define DSA_TN 32 /* rows scored per tile */

template <int D>
__global__ void d_gather_attn_decode(float* __restrict__ Opart, float* __restrict__ MLpart,
                                     const dsa_bf16* __restrict__ Q, const dsa_bf16* __restrict__ K,
                                     const dsa_bf16* __restrict__ V, const int* __restrict__ idx,
                                     const unsigned* __restrict__ n_sel, unsigned n_head,
                                     unsigned nsplit, float scale) {
    static_assert(D % 32 == 0, "D must be a whole number of 32-lane strips");
    __shared__ float s_sc[DSA_TN];
    __shared__ float s_q[D];

    const unsigned sp = blockIdx.x, h = blockIdx.y;
    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 31u, warp = tid >> 5, NW = D / 32u;
    const unsigned n = *n_sel;

    for (unsigned i = tid; i < (unsigned)D; i += D) s_q[i] = dsa_b2f(Q[(size_t)h * D + i]);
    __syncthreads();

    /* this split's contiguous slice of the selected set */
    const unsigned per = (n + nsplit - 1u) / nsplit;
    const unsigned lo = sp * per;
    const unsigned hi = (lo + per < n) ? (lo + per) : n;

    float m = -INFINITY, l = 0.0f, acc = 0.0f; /* thread `tid` owns output dim tid */

    for (unsigned r0 = lo; r0 < hi; r0 += DSA_TN) {
        const unsigned rn = (r0 + DSA_TN < hi) ? (unsigned)DSA_TN : (hi - r0);
        /* --- Phase A: scores for rows [r0, r0+rn) --- */
        for (unsigned r = warp; r < rn; r += NW) {
            const size_t row = (size_t)(unsigned)idx[r0 + r];
            float p = 0.0f;
            for (unsigned i = lane; i < (unsigned)D; i += 32u)
                p += s_q[i] * dsa_b2f(K[row * D + i]);
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) /* [W32] 5 steps for 32 lanes, not 6 */
                p += __shfl_xor_sync(0xFFFFFFFFu, p, off, 32);
            if (lane == 0) s_sc[r] = p * scale;
        }
        __syncthreads();

        /* --- Phase B: online-softmax rescale, then accumulate this tile's V --- */
        float tm = m;
        for (unsigned r = 0; r < rn; r++) tm = fmaxf(tm, s_sc[r]);
        const float corr = (m == -INFINITY) ? 0.0f : __expf(m - tm);
        acc *= corr;
        l *= corr;
        for (unsigned r = 0; r < rn; r++) {
            const float pr = __expf(s_sc[r] - tm);
            l += pr;
            acc += pr * dsa_b2f(V[(size_t)(unsigned)idx[r0 + r] * D + tid]);
        }
        m = tm;
        __syncthreads();
    }
    /* l was accumulated redundantly by all D threads (identical value); store once. */
    const size_t base = ((size_t)h * nsplit + sp);
    Opart[base * D + tid] = acc;
    if (tid == 0) { MLpart[base * 2 + 0] = m; MLpart[base * 2 + 1] = l; }
}

/* Combine the nsplit online-softmax partials. One block per head, D threads. */
template <int D>
__global__ void d_gather_merge(dsa_bf16* __restrict__ O, const float* __restrict__ Opart,
                               const float* __restrict__ MLpart, unsigned nsplit) {
    const unsigned h = blockIdx.x, tid = threadIdx.x;
    const float* ml = MLpart + (size_t)h * nsplit * 2;
    float gm = -INFINITY;
    for (unsigned s = 0; s < nsplit; s++) gm = fmaxf(gm, ml[s * 2]);
    float gl = 0.0f;
    for (unsigned s = 0; s < nsplit; s++) {
        if (ml[s * 2] == -INFINITY) continue;
        gl += ml[s * 2 + 1] * __expf(ml[s * 2] - gm);
    }
    const float inv = (gl > 0.0f) ? (1.0f / gl) : 0.0f;
    float a = 0.0f;
    for (unsigned s = 0; s < nsplit; s++) {
        if (ml[s * 2] == -INFINITY) continue;
        a += Opart[((size_t)h * nsplit + s) * D + tid] * __expf(ml[s * 2] - gm);
    }
    O[(size_t)h * D + tid] = __float2bfloat16(a * inv);
}

#endif /* PLOW_OP_DSA_CUH */
