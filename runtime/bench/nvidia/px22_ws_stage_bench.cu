/* px22_ws_stage_bench.cu — does decoupling cp.async ISSUE from the mma, INSIDE one op body,
 * beat the barrier-synchronised uniform loop on the production w8a8 [128][64] e4m3 tile?
 *
 * WHY. PX-9 attributed plow's 61-66%-of-fp8-peak w8a8 prefill GEMM to the cp.async staging path
 * by elimination (mainloop 94.3% of ceiling; cuBLASLt 95-99% on the same shapes; stage depth
 * worth 0%). PX-13 then killed TMA on this part (2-D `cp.async.bulk.tensor` measured 1.104x
 * SLOWER than the shipped `cp.async.cg`, byte-identical). What was never tested on sm_120a is the
 * OTHER half of the CUTLASS recipe: producer/consumer WARP SPECIALIZATION — the copies issued by
 * dedicated warps so the mma warps never stop issuing mma.
 *
 * THE ARITHMETIC THAT MOTIVATES IT. Per K-tile at BM=BN=128/BK8=64, 8 warps:
 *     mma      256 warp-QMMA/block, 8192 FLOP each, ceiling 1024 FLOP/clk/SM  -> 2048 cyc
 *     PX-9 rung 3 (mma + smem frags + barrier, no global)                     -> 2172 cyc
 *     PX-13 staging alone (1024 x cp.async.cg 16 B over 256 threads)          ->  819 cyc
 * If the two overlapped, a K-tile would cost ~2172 cyc. The shipped kernel runs at ~63% of the
 * 518.5 TFLOP/s ceiling, i.e. ~3250 cyc/K-tile ~= 2172 + 819 + change: the staging issue is
 * ADDITIVE, not overlapped. It cannot overlap, because EVERY warp issues the copies and then
 * waits at a `__syncthreads` before its own mma — there is no other warp left to run the mma
 * while the LSU works. That is the mechanism this file tests.
 *
 * WHAT IS MEASURED. Real bodies, real global memory, the production tile, the shipped swizzle and
 * fragment readers, the real epilogue. The instrument is SM cycles per staged+computed K-tile
 * (PX-9's unit: a clock that moves with power draw cannot corrupt a cycle ratio).
 *
 *   mma_only     stage the ring ONCE, then only mma from it. No global traffic, no wait.
 *                => the mainloop-only floor a perfectly-overlapped producer would deliver.
 *   stage_only   only staging + wait + barrier, no mma. => PX-13's 818.9 cyc/K-tile, re-measured
 *                with the production predication and swizzle in place.
 *   uniform      THE SHIPPED BODY, verbatim: 8 warps, 4x2 warp grid, acc[2][8][4], every warp
 *                stages 4 lines, cp.async.wait_group<NS-1>, two __syncthreads per K-tile.
 *   uniform_nb   uniform with both per-K-tile __syncthreads deleted. RACY — timing-only, its
 *                hash is expected to differ. Isolates the barrier bill inside the real body.
 *   ws4          WARP-SPECIALIZED, SAME 256 threads, SAME block, split INSIDE the body: warps
 *                4-7 produce (cp.async + `cp.async.mbarrier.arrive`), warps 0-3 consume with a
 *                2x2 warp grid (acc[4][8][4] = 128 f32/thread). No __syncthreads in either loop;
 *                the handshake is NS full/empty mbarrier pairs, so the producer runs ahead
 *                across K-tiles AND across output tiles.
 *   ws4b         same, consumer grid 4x1 (acc[2][16][4]) — same 128 f32/thread, different
 *                fragment mix (2 A-frags + 16 B-frags vs 4 + 8).
 *   ws4_iss2     ws4 but only 2 of the 4 producer warps issue copies: is 2 warps of LSU issue
 *                enough, i.e. how much of the block does the producer really need?
 *   ws4_clampE   ws4 under `__maxnreg__(E)` with NO setmaxnreg — separates the entry clamp from
 *                the donation (the control hopper_warpspec_prefill.cu showed is mandatory).
 *   ws4_smr      ws4 + `setmaxnreg.dec` on the producer warpgroup / `.inc` on the consumer one.
 *
 * WARP -> SUB-PARTITION. warp w is issued by SMSP w%4, so consumers 0-3 and producers 4-7 put
 * exactly one consumer and one producer on each of the 4 sub-partitions. That is the placement
 * warp specialization wants and it falls out of the split for free.
 *
 * BIT-EXACTNESS. Every arm accumulates output element (r,c) over the same k sequence in the same
 * order (ks ascending, sub-group kf=0 then kf=32, one m16n8k32 each) — only the WARP that owns
 * (r,c) changes. So `uniform`, `ws4`, `ws4b` and every register variant must produce a
 * BIT-IDENTICAL C. Checked with FNV-1a over the entire C plane (PX-13's gate).
 *
 * L2. B is 59 MB against a 96 MiB L2, and PX-9 measured cold vs warm within 0.5% on both engines
 * at these shapes, so no replication protocol here: this bench measures the ISSUE path.
 *
 * BUILD: perf-data/px22_build.sh      RUN: perf-data/tools/gpulease px22 /tmp/px22
 */
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define CK(x)                                                                                    \
    do {                                                                                         \
        cudaError_t e_ = (x);                                                                    \
        if (e_ != cudaSuccess) {                                                                 \
            printf("CUDA %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e_));                   \
            exit(1);                                                                             \
        }                                                                                        \
    } while (0)

/* ---- the production tile, verbatim from op_gemm.cuh ---- */
static const int BM = 128, BN = 128, BK8 = 64, THREADS = 256;
static const int ABUF = BM * BK8; /* 8192 B */
static const int BBUF = BN * BK8; /* 8192 B */
static const int MBAR_MAX = 2 * 8 * 8; /* NS<=8 full+empty mbarriers, 8 B each */

/* PGM_SW8_V2, the shipped default: XOR the 2-bit 16-byte-line slot with the low 2 row bits. */
__device__ __forceinline__ int sw8(int off) { return off ^ ((off >> 2) & 0x30); }

/* ---- primitives (op_gemm.cuh / hopper_warpspec_prefill.cu) ---- */
__device__ __forceinline__ void cp16(void* smem, const void* gmem, int src_bytes) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem),
                 "r"(src_bytes));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void mbar_init(uint64_t* b, int count) {
    unsigned a = (unsigned)__cvta_generic_to_shared(b);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(a), "r"(count));
}
__device__ __forceinline__ void mbar_arrive(uint64_t* b) {
    unsigned a = (unsigned)__cvta_generic_to_shared(b);
    asm volatile("{\n.reg .b64 st;\nmbarrier.arrive.shared::cta.b64 st, [%0];\n}\n" ::"r"(a));
}
/* arrive-on-completion of ALL prior cp.async issued by this thread; .noinc because the initial
 * expected-arrival count (= producer thread count) already covers this arrival. */
__device__ __forceinline__ void cp_mbar_arrive(uint64_t* b) {
    unsigned a = (unsigned)__cvta_generic_to_shared(b);
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(a));
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, int parity) {
    unsigned a = (unsigned)__cvta_generic_to_shared(b);
    asm volatile("{\n.reg .pred p;\nPXW%=:\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
                 "@!p bra PXW%=;\n}\n" ::"r"(a),
                 "r"(parity));
}
/* m16n8k32 e4m3 mma, f32 accumulate — pgm_mma_fp8_k32 verbatim. */
__device__ __forceinline__ void mma_fp8_k32(float (&d)[4], const unsigned (&a)[4],
                                            const unsigned (&b)[2], const float (&c)[4]) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]),
                   "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

/* pgm_stage_a8 / pgm_stage_b8 with the row count and the staging-thread count parameterised.
 * ROWS x 4 lines of 16 B, strided over NTHR threads; predication and swizzle as shipped. */
template <int ROWS, int NTHR>
__device__ __forceinline__ void stage_tile(uint8_t* dst, const uint8_t* __restrict__ src, int tid,
                                           int row0, int kbase, int R, int K) {
    const int LCH = BK8 / 16;
#pragma unroll
    for (int L = tid; L < ROWS * LCH; L += NTHR) {
        const int row = L / LCH, kk16 = (L % LCH) * 16;
        const int gr = row0 + row, gk = kbase + kk16;
        const bool in = (gr < R) && (gk + 16 <= K);
        const uint8_t* g = in ? src + (size_t)gr * K + gk : src;
        cp16(&dst[sw8(row * BK8 + kk16)], g, in ? 16 : 0);
    }
}

/* pgm_load_afrags_w8a8 / pgm_load_bfrags_w8a8 (PGM_W8A8_LDS64=1, the shipped default). */
template <int WM, int MFRAG>
__device__ __forceinline__ void load_afrags(unsigned (&af)[MFRAG][4], const uint8_t* Ad8, int wm,
                                            int kf, int lane) {
    const int kb = kf + 8 * (lane & 3);
#pragma unroll
    for (int mi = 0; mi < MFRAG; mi++) {
        const int rlo = wm * WM + mi * 16 + (lane >> 2);
        const uint2 lo = *(const uint2*)(Ad8 + sw8(rlo * BK8 + kb));
        const uint2 hi = *(const uint2*)(Ad8 + sw8((rlo + 8) * BK8 + kb));
        af[mi][0] = lo.x; af[mi][2] = lo.y;
        af[mi][1] = hi.x; af[mi][3] = hi.y;
    }
}
template <int WN, int NFRAG>
__device__ __forceinline__ void load_bfrags(unsigned (&bf)[NFRAG][2], const uint8_t* Bd8, int wn,
                                            int kf, int lane) {
    const int kb = kf + 8 * (lane & 3);
#pragma unroll
    for (int nj = 0; nj < NFRAG; nj++) {
        const int col = wn * WN + nj * 8 + (lane >> 2);
        const uint2 v = *(const uint2*)(Bd8 + sw8(col * BK8 + kb));
        bf[nj][0] = v.x; bf[nj][1] = v.y;
    }
}

struct Args {
    __nv_bfloat16* C;
    const uint8_t* A;
    const uint8_t* B;
    const uint8_t* B2; /* second weight stream (GLU 'up'); unused by the plain arms */
    const float* as;
    const float* ws;
    const float* ws2;
    int M, N, K;
    unsigned long long* cyc;
    unsigned* sink;
};

/* ================= UNIFORM: the shipped d_gemm_w8a8 body =================
 * BARS=false deletes the two per-K-tile __syncthreads (racy; timing-only). */
template <int NS, bool BARS>
__device__ __forceinline__ void body_uniform(Args g) {
    extern __shared__ __align__(16) uint8_t sm[];
    uint8_t* As = sm;
    uint8_t* Bs = sm + NS * ABUF;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp / 2, wn = warp % 2; /* PGM_WARPS_M=4, PGM_WARPS_N=2 */
    constexpr int WM = BM / 4, WN = BN / 2, MFRAG = WM / 16, NFRAG = WN / 8; /* 32,64,2,8 */
    const int tiles_n = (g.N + BN - 1) / BN, ntiles = ((g.M + BM - 1) / BM) * tiles_n;
    const int ksteps = (g.K + BK8 - 1) / BK8;

    long long t0 = clock64();
    for (int tile = (int)blockIdx.x; tile < ntiles; tile += (int)gridDim.x) {
        const int tm = (tile / tiles_n) * BM, tn = (tile % tiles_n) * BN;
        float acc[MFRAG][NFRAG][4];
#pragma unroll
        for (int i = 0; i < MFRAG; i++)
            for (int j = 0; j < NFRAG; j++)
                for (int e = 0; e < 4; e++) acc[i][j][e] = 0.f;

        auto stage = [&](int ks, int buf) {
            stage_tile<BM, THREADS>(As + buf * ABUF, g.A, tid, tm, ks * BK8, g.M, g.K);
            stage_tile<BN, THREADS>(Bs + buf * BBUF, g.B, tid, tn, ks * BK8, g.N, g.K);
        };
#pragma unroll
        for (int s = 0; s < NS - 1; s++) {
            if (s < ksteps) stage(s, s);
            cp_commit();
        }
        for (int ks = 0; ks < ksteps; ks++) {
            const int fetch = ks + NS - 1;
            if (fetch < ksteps) stage(fetch, fetch % NS);
            cp_commit();
            cp_wait<NS - 1>();
            if (BARS) __syncthreads();
            const int cb = ks % NS;
#pragma unroll
            for (int kf = 0; kf < BK8; kf += 32) {
                unsigned af[MFRAG][4];
                load_afrags<WM, MFRAG>(af, As + cb * ABUF, wm, kf, lane);
                unsigned bf[NFRAG][2];
                load_bfrags<WN, NFRAG>(bf, Bs + cb * BBUF, wn, kf, lane);
#pragma unroll
                for (int mi = 0; mi < MFRAG; mi++)
#pragma unroll
                    for (int nj = 0; nj < NFRAG; nj++)
                        mma_fp8_k32(acc[mi][nj], af[mi], bf[nj], acc[mi][nj]);
            }
            if (BARS) __syncthreads();
        }
#pragma unroll
        for (int mi = 0; mi < MFRAG; mi++)
#pragma unroll
            for (int nj = 0; nj < NFRAG; nj++) {
                const int gr = wm * WM + mi * 16 + (lane / 4);
                const int gc = wn * WN + nj * 8 + (lane % 4) * 2;
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int rr = tm + gr + (e / 2) * 8, cc = tn + gc + (e % 2);
                    if (rr < g.M && cc < g.N)
                        g.C[(size_t)rr * g.N + cc] =
                            __float2bfloat16(acc[mi][nj][e] * g.as[rr] * g.ws[cc]);
                }
            }
        __syncthreads();
    }
    if (tid == 0) g.cyc[blockIdx.x] = (unsigned long long)(clock64() - t0);
}

/* ================= reference floors ================= */
/* mma_only: fill the whole ring once, then run ONLY the mainloop out of it.
 * BARS=true keeps the per-K-tile __syncthreads (PX-9 rung 3); false removes it (rung 1).
 * CWM x CWN is the warp grid and NW = CWM*CWN the warps that run mma — NW=4/2x2 reproduces the
 * WARP-SPECIALIZED consumer's exact fragment shape with no staging at all, which is what
 * separates "the copies are decoupled" from "the consumer's warp tile is twice as tall". */
template <int NS, bool BARS, int CWM, int CWN> __device__ __forceinline__ void body_mma_only(Args g) {
    extern __shared__ __align__(16) uint8_t sm[];
    uint8_t* As = sm;
    uint8_t* Bs = sm + NS * ABUF;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    constexpr int NW = CWM * CWN;
    const int wm = warp / CWN, wn = warp % CWN;
    constexpr int WM = BM / CWM, WN = BN / CWN, MFRAG = WM / 16, NFRAG = WN / 8;
    const int tiles_n = (g.N + BN - 1) / BN, ntiles = ((g.M + BM - 1) / BM) * tiles_n;
    const int ksteps = (g.K + BK8 - 1) / BK8;
#pragma unroll
    for (int s = 0; s < NS; s++) {
        stage_tile<BM, THREADS>(As + s * ABUF, g.A, tid, 0, s * BK8, g.M, g.K);
        stage_tile<BN, THREADS>(Bs + s * BBUF, g.B, tid, 0, s * BK8, g.N, g.K);
    }
    cp_commit();
    cp_wait<0>();
    __syncthreads();
    if (warp >= NW) return; /* the non-participating warps occupy their slots and issue nothing */

    long long t0 = clock64();
    for (int tile = (int)blockIdx.x; tile < ntiles; tile += (int)gridDim.x) {
        const int tm = (tile / tiles_n) * BM, tn = (tile % tiles_n) * BN;
        float acc[MFRAG][NFRAG][4];
#pragma unroll
        for (int i = 0; i < MFRAG; i++)
            for (int j = 0; j < NFRAG; j++)
                for (int e = 0; e < 4; e++) acc[i][j][e] = 0.f;
        for (int ks = 0; ks < ksteps; ks++) {
            const int cb = ks % NS;
            if (BARS) __syncthreads(); /* PX-9 rung 3 keeps it, rung 1 does not */
#pragma unroll
            for (int kf = 0; kf < BK8; kf += 32) {
                unsigned af[MFRAG][4];
                load_afrags<WM, MFRAG>(af, As + cb * ABUF, wm, kf, lane);
                unsigned bf[NFRAG][2];
                load_bfrags<WN, NFRAG>(bf, Bs + cb * BBUF, wn, kf, lane);
#pragma unroll
                for (int mi = 0; mi < MFRAG; mi++)
#pragma unroll
                    for (int nj = 0; nj < NFRAG; nj++)
                        mma_fp8_k32(acc[mi][nj], af[mi], bf[nj], acc[mi][nj]);
            }
        }
#pragma unroll
        for (int mi = 0; mi < MFRAG; mi++)
#pragma unroll
            for (int nj = 0; nj < NFRAG; nj++) {
                const int gr = wm * WM + mi * 16 + (lane / 4);
                const int gc = wn * WN + nj * 8 + (lane % 4) * 2;
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int rr = tm + gr + (e / 2) * 8, cc = tn + gc + (e % 2);
                    if (rr < g.M && cc < g.N)
                        g.C[(size_t)rr * g.N + cc] =
                            __float2bfloat16(acc[mi][nj][e] * g.as[rr] * g.ws[cc]);
                }
            }
        __syncthreads();
    }
    if (tid == 0) g.cyc[blockIdx.x] = (unsigned long long)(clock64() - t0);
}

/* stage_only: the staging path with the mainloop removed (PX-13's arm, production predication). */
template <int NS> __device__ __forceinline__ void body_stage_only(Args g) {
    extern __shared__ __align__(16) uint8_t sm[];
    uint8_t* As = sm;
    uint8_t* Bs = sm + NS * ABUF;
    const int tid = threadIdx.x;
    const int tiles_n = (g.N + BN - 1) / BN, ntiles = ((g.M + BM - 1) / BM) * tiles_n;
    const int ksteps = (g.K + BK8 - 1) / BK8;
    long long t0 = clock64();
    for (int tile = (int)blockIdx.x; tile < ntiles; tile += (int)gridDim.x) {
        const int tm = (tile / tiles_n) * BM, tn = (tile % tiles_n) * BN;
        auto stage = [&](int ks, int buf) {
            stage_tile<BM, THREADS>(As + buf * ABUF, g.A, tid, tm, ks * BK8, g.M, g.K);
            stage_tile<BN, THREADS>(Bs + buf * BBUF, g.B, tid, tn, ks * BK8, g.N, g.K);
        };
#pragma unroll
        for (int s = 0; s < NS - 1; s++) {
            if (s < ksteps) stage(s, s);
            cp_commit();
        }
        for (int ks = 0; ks < ksteps; ks++) {
            const int fetch = ks + NS - 1;
            if (fetch < ksteps) stage(fetch, fetch % NS);
            cp_commit();
            cp_wait<NS - 1>();
            __syncthreads();
            if (As[(ks % NS) * ABUF + tid] == 0xffu) g.sink[0]++;
            __syncthreads();
        }
    }
    if (tid == 0) g.cyc[blockIdx.x] = (unsigned long long)(clock64() - t0);
}

/* ================= WARP-SPECIALIZED body, split INSIDE the op body =================
 * NPROD producer warps (the HIGH warp ids, so consumers keep warps 0..NCONS-1 and both groups
 * spread over all 4 sub-partitions), NISS of them actually issuing. CWM x CWN consumer warp grid.
 * SMR emits setmaxnreg dec/inc — legal only when both groups are 128-thread aligned. */
template <int NS, int NPROD, int NISS, int CWM, int CWN, bool SMR, int PREG, int CREG>
__device__ __forceinline__ void body_ws(Args g) {
    constexpr int NCONS = 8 - NPROD;
    constexpr int WM = BM / CWM, WN = BN / CWN, MFRAG = WM / 16, NFRAG = WN / 8;
    static_assert(CWM * CWN == NCONS, "consumer warp grid must cover the consumer warps");
    static_assert(WM % 16 == 0 && WN % 8 == 0, "warp tile must be whole fragments");
    extern __shared__ __align__(16) uint8_t sm[];
    uint64_t* bfull = (uint64_t*)sm;
    uint64_t* bempty = bfull + NS;
    uint8_t* As = sm + MBAR_MAX;
    uint8_t* Bs = As + NS * ABUF;

    const int tid = threadIdx.x;
    if (tid < NS) {
        mbar_init(bfull + tid, NPROD * 32);
        mbar_init(bempty + tid, NCONS * 32);
    }
    __syncthreads(); /* the only block-wide barrier; both branches are __syncthreads-free after */

    const int tiles_n = (g.N + BN - 1) / BN, ntiles = ((g.M + BM - 1) / BM) * tiles_n;
    const int ksteps = (g.K + BK8 - 1) / BK8;
    long long t0 = clock64();

    if (tid >= NCONS * 32) {
        /* ---------------- producer warps ---------------- */
        if (SMR) asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(PREG));
        const int ptid = tid - NCONS * 32;
        int st = 0;
        for (int tile = (int)blockIdx.x; tile < ntiles; tile += (int)gridDim.x) {
            const int tm = (tile / tiles_n) * BM, tn = (tile % tiles_n) * BN;
            for (int ks = 0; ks < ksteps; ks++, st++) {
                const int s = st % NS;
                if (st >= NS) mbar_wait(bempty + s, ((st / NS) + 1) & 1);
                if (ptid < NISS * 32) {
                    stage_tile<BM, NISS * 32>(As + s * ABUF, g.A, ptid, tm, ks * BK8, g.M, g.K);
                    stage_tile<BN, NISS * 32>(Bs + s * BBUF, g.B, ptid, tn, ks * BK8, g.N, g.K);
                }
                cp_mbar_arrive(bfull + s); /* arrives when THIS thread's copies have landed */
            }
        }
    } else {
        /* ---------------- consumer warps ---------------- */
        if (SMR) asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(CREG));
        const int warp = tid >> 5, lane = tid & 31;
        const int wm = warp / CWN, wn = warp % CWN;
        int st = 0;
        for (int tile = (int)blockIdx.x; tile < ntiles; tile += (int)gridDim.x) {
            const int tm = (tile / tiles_n) * BM, tn = (tile % tiles_n) * BN;
            float acc[MFRAG][NFRAG][4];
#pragma unroll
            for (int i = 0; i < MFRAG; i++)
                for (int j = 0; j < NFRAG; j++)
                    for (int e = 0; e < 4; e++) acc[i][j][e] = 0.f;
            for (int ks = 0; ks < ksteps; ks++, st++) {
                const int s = st % NS;
                mbar_wait(bfull + s, (st / NS) & 1);
#pragma unroll
                for (int kf = 0; kf < BK8; kf += 32) {
                    unsigned af[MFRAG][4];
                    load_afrags<WM, MFRAG>(af, As + s * ABUF, wm, kf, lane);
                    unsigned bf[NFRAG][2];
                    load_bfrags<WN, NFRAG>(bf, Bs + s * BBUF, wn, kf, lane);
#pragma unroll
                    for (int mi = 0; mi < MFRAG; mi++)
#pragma unroll
                        for (int nj = 0; nj < NFRAG; nj++)
                            mma_fp8_k32(acc[mi][nj], af[mi], bf[nj], acc[mi][nj]);
                }
                /* the mma consumed af/bf from registers, so every LDS has already returned */
                mbar_arrive(bempty + s);
            }
#pragma unroll
            for (int mi = 0; mi < MFRAG; mi++)
#pragma unroll
                for (int nj = 0; nj < NFRAG; nj++) {
                    const int gr = wm * WM + mi * 16 + (lane / 4);
                    const int gc = wn * WN + nj * 8 + (lane % 4) * 2;
#pragma unroll
                    for (int e = 0; e < 4; e++) {
                        const int rr = tm + gr + (e / 2) * 8, cc = tn + gc + (e % 2);
                        if (rr < g.M && cc < g.N)
                            g.C[(size_t)rr * g.N + cc] =
                                __float2bfloat16(acc[mi][nj][e] * g.as[rr] * g.ws[cc]);
                    }
                }
        }
    }
    if (tid == 0) g.cyc[blockIdx.x] = (unsigned long long)(clock64() - t0);
}

/* ================= GLU: the register-limited arm, ~2/3 of prefill GEMM FLOPs =================
 * d_gemm_glu_w8a8 stages ONE A tile and TWO weight tiles per K-tile and holds TWO accumulator
 * sets. At the shipped GLU_BN=128 with 8 warps that is already 128 f32/thread. Splitting 4 warps
 * off to produce would double the accumulators per consumer to 256 f32/thread — impossible — so
 * the warp-specialized GLU MUST also halve the N-tile. GBN is therefore a template parameter and
 * the uniform arm is measured at BOTH widths so the two changes can be separated.
 * silu(g)*u, both streams row-scaled by ascale — the shipped epilogue. */
__device__ __forceinline__ float silu(float x) { return x / (1.f + __expf(-x)); }

template <int GBN, int NS> __device__ __forceinline__ void body_glu_uniform(Args g) {
    extern __shared__ __align__(16) uint8_t sm[];
    constexpr int GBBUF = GBN * BK8;
    uint8_t* As = sm;
    uint8_t* Bg = sm + NS * ABUF;
    uint8_t* Bu = Bg + NS * GBBUF;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp / 2, wn = warp % 2;
    constexpr int WM = BM / 4, WN = GBN / 2, MFRAG = WM / 16, NFRAG = WN / 8;
    const int tiles_n = (g.N + GBN - 1) / GBN, ntiles = ((g.M + BM - 1) / BM) * tiles_n;
    const int ksteps = (g.K + BK8 - 1) / BK8;

    long long t0 = clock64();
    for (int tile = (int)blockIdx.x; tile < ntiles; tile += (int)gridDim.x) {
        const int tm = (tile / tiles_n) * BM, tn = (tile % tiles_n) * GBN;
        float accg[MFRAG][NFRAG][4], accu[MFRAG][NFRAG][4];
#pragma unroll
        for (int i = 0; i < MFRAG; i++)
            for (int j = 0; j < NFRAG; j++)
                for (int e = 0; e < 4; e++) { accg[i][j][e] = 0.f; accu[i][j][e] = 0.f; }
        auto stage = [&](int ks, int buf) {
            stage_tile<BM, THREADS>(As + buf * ABUF, g.A, tid, tm, ks * BK8, g.M, g.K);
            stage_tile<GBN, THREADS>(Bg + buf * GBBUF, g.B, tid, tn, ks * BK8, g.N, g.K);
            stage_tile<GBN, THREADS>(Bu + buf * GBBUF, g.B2, tid, tn, ks * BK8, g.N, g.K);
        };
#pragma unroll
        for (int s = 0; s < NS - 1; s++) {
            if (s < ksteps) stage(s, s);
            cp_commit();
        }
        for (int ks = 0; ks < ksteps; ks++) {
            const int fetch = ks + NS - 1;
            if (fetch < ksteps) stage(fetch, fetch % NS);
            cp_commit();
            cp_wait<NS - 1>();
            __syncthreads();
            const int cb = ks % NS;
#pragma unroll
            for (int kf = 0; kf < BK8; kf += 32) {
                unsigned af[MFRAG][4];
                load_afrags<WM, MFRAG>(af, As + cb * ABUF, wm, kf, lane);
                unsigned bg[NFRAG][2], bu[NFRAG][2];
                load_bfrags<WN, NFRAG>(bg, Bg + cb * GBBUF, wn, kf, lane);
                load_bfrags<WN, NFRAG>(bu, Bu + cb * GBBUF, wn, kf, lane);
#pragma unroll
                for (int mi = 0; mi < MFRAG; mi++)
#pragma unroll
                    for (int nj = 0; nj < NFRAG; nj++) {
                        mma_fp8_k32(accg[mi][nj], af[mi], bg[nj], accg[mi][nj]);
                        mma_fp8_k32(accu[mi][nj], af[mi], bu[nj], accu[mi][nj]);
                    }
            }
            __syncthreads();
        }
#pragma unroll
        for (int mi = 0; mi < MFRAG; mi++)
#pragma unroll
            for (int nj = 0; nj < NFRAG; nj++) {
                const int gr = wm * WM + mi * 16 + (lane / 4);
                const int gc = wn * WN + nj * 8 + (lane % 4) * 2;
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int rr = tm + gr + (e / 2) * 8, cc = tn + gc + (e % 2);
                    if (rr < g.M && cc < g.N) {
                        const float a = g.as[rr];
                        g.C[(size_t)rr * g.N + cc] = __float2bfloat16(
                            silu(accg[mi][nj][e] * a * g.ws[cc]) * (accu[mi][nj][e] * a * g.ws2[cc]));
                    }
                }
            }
        __syncthreads();
    }
    if (tid == 0) g.cyc[blockIdx.x] = (unsigned long long)(clock64() - t0);
}

template <int GBN, int NS, int NPROD, int CWM, int CWN>
__device__ __forceinline__ void body_glu_ws(Args g) {
    constexpr int NCONS = 8 - NPROD;
    constexpr int GBBUF = GBN * BK8;
    constexpr int WM = BM / CWM, WN = GBN / CWN, MFRAG = WM / 16, NFRAG = WN / 8;
    static_assert(CWM * CWN == NCONS, "consumer warp grid must cover the consumer warps");
    extern __shared__ __align__(16) uint8_t sm[];
    uint64_t* bfull = (uint64_t*)sm;
    uint64_t* bempty = bfull + NS;
    uint8_t* As = sm + MBAR_MAX;
    uint8_t* Bg = As + NS * ABUF;
    uint8_t* Bu = Bg + NS * GBBUF;

    const int tid = threadIdx.x;
    if (tid < NS) {
        mbar_init(bfull + tid, NPROD * 32);
        mbar_init(bempty + tid, NCONS * 32);
    }
    __syncthreads();
    const int tiles_n = (g.N + GBN - 1) / GBN, ntiles = ((g.M + BM - 1) / BM) * tiles_n;
    const int ksteps = (g.K + BK8 - 1) / BK8;
    long long t0 = clock64();

    if (tid >= NCONS * 32) {
        const int ptid = tid - NCONS * 32;
        int st = 0;
        for (int tile = (int)blockIdx.x; tile < ntiles; tile += (int)gridDim.x) {
            const int tm = (tile / tiles_n) * BM, tn = (tile % tiles_n) * GBN;
            for (int ks = 0; ks < ksteps; ks++, st++) {
                const int s = st % NS;
                if (st >= NS) mbar_wait(bempty + s, ((st / NS) + 1) & 1);
                stage_tile<BM, NPROD * 32>(As + s * ABUF, g.A, ptid, tm, ks * BK8, g.M, g.K);
                stage_tile<GBN, NPROD * 32>(Bg + s * GBBUF, g.B, ptid, tn, ks * BK8, g.N, g.K);
                stage_tile<GBN, NPROD * 32>(Bu + s * GBBUF, g.B2, ptid, tn, ks * BK8, g.N, g.K);
                cp_mbar_arrive(bfull + s);
            }
        }
    } else {
        const int warp = tid >> 5, lane = tid & 31;
        const int wm = warp / CWN, wn = warp % CWN;
        int st = 0;
        for (int tile = (int)blockIdx.x; tile < ntiles; tile += (int)gridDim.x) {
            const int tm = (tile / tiles_n) * BM, tn = (tile % tiles_n) * GBN;
            float accg[MFRAG][NFRAG][4], accu[MFRAG][NFRAG][4];
#pragma unroll
            for (int i = 0; i < MFRAG; i++)
                for (int j = 0; j < NFRAG; j++)
                    for (int e = 0; e < 4; e++) { accg[i][j][e] = 0.f; accu[i][j][e] = 0.f; }
            for (int ks = 0; ks < ksteps; ks++, st++) {
                const int s = st % NS;
                mbar_wait(bfull + s, (st / NS) & 1);
#pragma unroll
                for (int kf = 0; kf < BK8; kf += 32) {
                    unsigned af[MFRAG][4];
                    load_afrags<WM, MFRAG>(af, As + s * ABUF, wm, kf, lane);
                    unsigned bg[NFRAG][2], bu[NFRAG][2];
                    load_bfrags<WN, NFRAG>(bg, Bg + s * GBBUF, wn, kf, lane);
                    load_bfrags<WN, NFRAG>(bu, Bu + s * GBBUF, wn, kf, lane);
#pragma unroll
                    for (int mi = 0; mi < MFRAG; mi++)
#pragma unroll
                        for (int nj = 0; nj < NFRAG; nj++) {
                            mma_fp8_k32(accg[mi][nj], af[mi], bg[nj], accg[mi][nj]);
                            mma_fp8_k32(accu[mi][nj], af[mi], bu[nj], accu[mi][nj]);
                        }
                }
                mbar_arrive(bempty + s);
            }
#pragma unroll
            for (int mi = 0; mi < MFRAG; mi++)
#pragma unroll
                for (int nj = 0; nj < NFRAG; nj++) {
                    const int gr = wm * WM + mi * 16 + (lane / 4);
                    const int gc = wn * WN + nj * 8 + (lane % 4) * 2;
#pragma unroll
                    for (int e = 0; e < 4; e++) {
                        const int rr = tm + gr + (e / 2) * 8, cc = tn + gc + (e % 2);
                        if (rr < g.M && cc < g.N) {
                            const float a = g.as[rr];
                            g.C[(size_t)rr * g.N + cc] = __float2bfloat16(
                                silu(accg[mi][nj][e] * a * g.ws[cc]) *
                                (accu[mi][nj][e] * a * g.ws2[cc]));
                        }
                    }
                }
        }
    }
    if (tid == 0) g.cyc[blockIdx.x] = (unsigned long long)(clock64() - t0);
}

/* ================= arm instantiation ================= */
#define ARM_LB(name, body)                                                                       \
    __global__ void __launch_bounds__(THREADS, 1) name(Args g) { body; }
#define ARM_MNR(name, E, body)                                                                    \
    __global__ void __maxnreg__(E) name(Args g) { body; }

ARM_LB(k_mma_only, (body_mma_only<3, true, 4, 2>(g)))
ARM_LB(k_mma_only_nb, (body_mma_only<3, false, 4, 2>(g)))
ARM_LB(k_mma_only_4w, (body_mma_only<3, false, 2, 2>(g)))
ARM_LB(k_stage_only, body_stage_only<3>(g))
ARM_LB(k_uniform3, (body_uniform<3, true>(g)))
ARM_LB(k_uniform4, (body_uniform<4, true>(g)))
ARM_LB(k_uniform3_nb, (body_uniform<3, false>(g)))
ARM_LB(k_ws4_s3, (body_ws<3, 4, 4, 2, 2, false, 0, 0>(g)))
ARM_LB(k_ws4_s4, (body_ws<4, 4, 4, 2, 2, false, 0, 0>(g)))
ARM_LB(k_ws4_s5, (body_ws<5, 4, 4, 2, 2, false, 0, 0>(g)))
ARM_LB(k_ws4_s6, (body_ws<6, 4, 4, 2, 2, false, 0, 0>(g)))
ARM_LB(k_ws4b_s4, (body_ws<4, 4, 4, 4, 1, false, 0, 0>(g)))
ARM_LB(k_ws4_iss2_s4, (body_ws<4, 4, 2, 2, 2, false, 0, 0>(g)))
ARM_LB(k_ws4_iss1_s4, (body_ws<4, 4, 1, 2, 2, false, 0, 0>(g)))
/* register arms: entry clamp alone (control) vs clamp + donation. PROD+CONS == 2*ENTRY. */
ARM_MNR(k_ws4_clamp168, 168, (body_ws<4, 4, 4, 2, 2, false, 0, 0>(g)))
ARM_MNR(k_ws4_smr168, 168, (body_ws<4, 4, 4, 2, 2, true, 88, 248>(g)))
ARM_MNR(k_ws4_clamp128, 128, (body_ws<4, 4, 4, 2, 2, false, 0, 0>(g)))
ARM_MNR(k_ws4_smr128, 128, (body_ws<4, 4, 4, 2, 2, true, 24, 232>(g)))
ARM_MNR(k_ws4_smr128b, 128, (body_ws<4, 4, 4, 2, 2, true, 88, 168>(g)))
/* GLU: shipped is GLU_BN=128 / GLU_STAGES=2. The ws consumer must halve the N-tile to keep the
 * two accumulator sets at 128 f32/thread, so GLU_BN=64 uniform is the matched control. */
ARM_LB(k_glu_u128_s2, (body_glu_uniform<128, 2>(g)))
ARM_LB(k_glu_u128_s3, (body_glu_uniform<128, 3>(g)))
ARM_LB(k_glu_u64_s2, (body_glu_uniform<64, 2>(g)))
ARM_LB(k_glu_u64_s4, (body_glu_uniform<64, 4>(g)))
ARM_LB(k_glu_ws64_s3, (body_glu_ws<64, 3, 4, 2, 2>(g)))
ARM_LB(k_glu_ws64_s4, (body_glu_ws<64, 4, 4, 2, 2>(g)))
ARM_LB(k_glu_ws64_s6, (body_glu_ws<64, 6, 4, 2, 2>(g)))

/* ================= host ================= */
static uint32_t rngs = 12345u;
static uint32_t xr() { rngs = rngs * 1664525u + 1013904223u; return rngs; }

/* BUG FOUND MID-RUN: the first version filled the operands with `xr() & 0x7f`, and 0x7f is NaN in
 * E4M3 (exponent 1111 + mantissa 111 — the only NaN encoding, and `& 0x7f` hits it 1/128 of the
 * time). One NaN anywhere in a K=3840 reduction makes the whole C plane NaN, every arm hashes the
 * same, and the FNV bit-exactness gate passes VACUOUSLY. Restricting the exponent to 5..9 keeps
 * every operand finite in [0.25, 7.5] and makes the hash actually discriminating. */
static uint8_t rnd_e4m3() {
    const uint32_t r = xr();
    return (uint8_t)((((r >> 16) & 1u) << 7) | ((5u + ((r >> 8) % 5u)) << 3) | (r & 7u));
}

static uint64_t fnv1a(const void* p, size_t n) {
    const uint8_t* b = (const uint8_t*)p;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

struct Res { double cpt, ms, tflops, ghz; uint64_t hash; int regs, spill; };

typedef void (*Kern)(Args);

/* TILEN = the arm's N-tile width (cyc/K-tile is only comparable within one tile shape).
 * FLOPX = GEMMs per output element (1 plain, 2 GLU); 0 suppresses the TFLOP/s column. */
static Res run(const char* label, Kern k, int smem, int G, Args g, int M, int N, int K,
               int reps, bool hash_it, int TILEN = BN, int FLOPX = 1) {
    Res r{};
    CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    cudaFuncAttributes fa;
    CK(cudaFuncGetAttributes(&fa, (const void*)k));
    r.regs = fa.numRegs;
    r.spill = (int)fa.localSizeBytes;

    CK(cudaMemset(g.C, 0, (size_t)M * N * sizeof(__nv_bfloat16)));
    k<<<G, THREADS, smem>>>(g);
    CK(cudaDeviceSynchronize());
    CK(cudaGetLastError());
    if (hash_it) {
        std::vector<__nv_bfloat16> h((size_t)M * N);
        CK(cudaMemcpy(h.data(), g.C, h.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        r.hash = fnv1a(h.data(), h.size() * sizeof(__nv_bfloat16));
    }
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0));
    for (int i = 0; i < reps; i++) k<<<G, THREADS, smem>>>(g);
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
    CK(cudaGetLastError());
    std::vector<unsigned long long> hc(G);
    CK(cudaMemcpy(hc.data(), g.cyc, sizeof(unsigned long long) * G, cudaMemcpyDeviceToHost));
    double mean = 0; for (int i = 0; i < G; i++) mean += (double)hc[i];
    mean /= G;
    const int tiles_n = (N + TILEN - 1) / TILEN, ntiles = ((M + BM - 1) / BM) * tiles_n;
    const double tiles_per_blk = (double)ntiles / G;
    const double ksteps = (K + BK8 - 1) / BK8;
    r.ms = ms / reps;
    r.cpt = mean / (tiles_per_blk * ksteps);
    r.ghz = mean / (r.ms * 1e-3) / 1e9;
    r.tflops = FLOPX ? FLOPX * 2.0 * M * N * K / (r.ms * 1e-3) / 1e12 : 0.0;
    printf("  %-14s %8.3f ms %9.1f cyc/K-tile %7.1f TF/s %6.3f GHz  regs=%3d spill=%4d  %016llx\n",
           label, r.ms, r.cpt, r.tflops, r.ghz, r.regs, r.spill, (unsigned long long)r.hash);
    if (r.tflops > 518.5) printf("      !! %.1f TF/s exceeds the 518.5 fp8 ceiling — bad denominator\n", r.tflops);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return r;
}

int main(int argc, char** argv) {
    CK(cudaFree(nullptr));
    cudaDeviceProp pr;
    CK(cudaGetDeviceProperties(&pr, 0));
    /* gate|up at the real prefill bucket: M=1024 activations, N=15360, K=3840. */
    const int M = getenv("PX22_M") ? atoi(getenv("PX22_M")) : 1024;
    const int N = 15360;
    /* PX22_K raises K only to amortize the per-OUTPUT-TILE cost (acc init + epilogue) over more
     * K-tiles: if cyc/K-tile falls with K, the in-context mma floor is inflated by the epilogue,
     * not by the mainloop. K must be a multiple of BK8. */
    const int K = getenv("PX22_K") ? atoi(getenv("PX22_K")) : 3840;
    const int reps = getenv("PX22_REPS") ? atoi(getenv("PX22_REPS")) : 20;
    const int tiles_n = N / BN, ntiles = (M / BM) * tiles_n;
    int G = pr.multiProcessorCount;
    while (ntiles % G) G--; /* even tiles/block: no wave quantization inside the cycle counter */
    printf("# %s SMs=%d  M=%d N=%d K=%d  tile %dx%dx%d  grid=%d (%d tiles, %d/blk)  reps=%d\n",
           pr.name, pr.multiProcessorCount, M, N, K, BM, BN, BK8, G, ntiles, ntiles / G, reps);
    printf("# mma floor at 1024 FLOP/clk/SM = %d cyc/K-tile; PX-9 rung3 = 2172; PX-13 staging = 818.9\n",
           2 * BM * BN * BK8 / 1024);

    uint8_t *A, *B, *B2;
    CK(cudaMalloc(&A, (size_t)M * K));
    CK(cudaMalloc(&B, (size_t)N * K));
    CK(cudaMalloc(&B2, (size_t)N * K));
    {
        std::vector<uint8_t> h((size_t)N * K);
        for (size_t i = 0; i < (size_t)M * K; i++) h[i] = rnd_e4m3();
        CK(cudaMemcpy(A, h.data(), (size_t)M * K, cudaMemcpyHostToDevice));
        for (size_t i = 0; i < (size_t)N * K; i++) h[i] = rnd_e4m3();
        CK(cudaMemcpy(B, h.data(), (size_t)N * K, cudaMemcpyHostToDevice));
        for (size_t i = 0; i < (size_t)N * K; i++) h[i] = rnd_e4m3();
        CK(cudaMemcpy(B2, h.data(), (size_t)N * K, cudaMemcpyHostToDevice));
    }
    float *as, *ws, *ws2;
    CK(cudaMalloc(&as, M * sizeof(float)));
    CK(cudaMalloc(&ws, N * sizeof(float)));
    CK(cudaMalloc(&ws2, N * sizeof(float)));
    {
        std::vector<float> h(N > M ? N : M);
        for (int i = 0; i < M; i++) h[i] = 0.01f + (float)(xr() & 0xff) * 1e-4f;
        CK(cudaMemcpy(as, h.data(), M * sizeof(float), cudaMemcpyHostToDevice));
        for (int i = 0; i < N; i++) h[i] = 0.01f + (float)(xr() & 0xff) * 1e-4f;
        CK(cudaMemcpy(ws, h.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        for (int i = 0; i < N; i++) h[i] = 0.01f + (float)(xr() & 0xff) * 1e-4f;
        CK(cudaMemcpy(ws2, h.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    }
    __nv_bfloat16* C;
    CK(cudaMalloc(&C, (size_t)M * N * sizeof(__nv_bfloat16)));
    unsigned long long* cyc;
    CK(cudaMalloc(&cyc, sizeof(unsigned long long) * pr.multiProcessorCount));
    unsigned* sink;
    CK(cudaMalloc(&sink, 4));
    CK(cudaMemset(sink, 0, 4));
    Args g{C, A, B, B2, as, ws, ws2, M, N, K, cyc, sink};

    const int SM3 = 3 * (ABUF + BBUF), SM4 = 4 * (ABUF + BBUF);
    const int W3 = MBAR_MAX + SM3, W4 = MBAR_MAX + SM4;
    const int W5 = MBAR_MAX + 5 * (ABUF + BBUF), W6 = MBAR_MAX + 6 * (ABUF + BBUF);
    printf("# smem: uniform NS=3 %d B, ws NS=3 %d B, NS=4 %d B, NS=5 %d B, NS=6 %d B (cap 101376)\n",
           SM3, W3, W4, W5, W6);

    /* the hash of the zeroed plane: any arm matching this produced nothing, and the whole
     * bit-exactness gate would be vacuous (see rnd_e4m3 — that exact failure happened once). */
    uint64_t zhash;
    {
        CK(cudaMemset(C, 0, (size_t)M * N * sizeof(__nv_bfloat16)));
        std::vector<__nv_bfloat16> h((size_t)M * N);
        CK(cudaMemcpy(h.data(), C, h.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        zhash = fnv1a(h.data(), h.size() * sizeof(__nv_bfloat16));
        printf("# zero-plane FNV %016llx — no arm may equal this\n", (unsigned long long)zhash);
    }

    printf("[floors] mma with no global traffic, and staging with no mma\n");
    run("mma_only", k_mma_only, SM3, G, g, M, N, K, reps, false);
    run("mma_only nobar", k_mma_only_nb, SM3, G, g, M, N, K, reps, false);
    run("mma_only 4warp", k_mma_only_4w, SM3, G, g, M, N, K, reps, false);
    run("stage_only", k_stage_only, SM3, G, g, M, N, K, reps, false, BN, 0);

    printf("[uniform] the shipped body\n");
    Res u = run("uniform NS=3", k_uniform3, SM3, G, g, M, N, K, reps, true);
    run("uniform NS=4", k_uniform4, SM4, G, g, M, N, K, reps, true);
    run("uniform nobar", k_uniform3_nb, SM3, G, g, M, N, K, reps, false);

    printf("[warp-specialized] 4 producer warps + 4 consumer warps, same 256 threads\n");
    Res w[12];
    int nw = 0;
    w[nw++] = run("ws4 NS=3", k_ws4_s3, W3, G, g, M, N, K, reps, true);
    w[nw++] = run("ws4 NS=4", k_ws4_s4, W4, G, g, M, N, K, reps, true);
    w[nw++] = run("ws4 NS=5", k_ws4_s5, W5, G, g, M, N, K, reps, true);
    w[nw++] = run("ws4 NS=6", k_ws4_s6, W6, G, g, M, N, K, reps, true);
    w[nw++] = run("ws4b 4x1 NS=4", k_ws4b_s4, W4, G, g, M, N, K, reps, true);
    w[nw++] = run("ws4 iss2 NS=4", k_ws4_iss2_s4, W4, G, g, M, N, K, reps, true);
    w[nw++] = run("ws4 iss1 NS=4", k_ws4_iss1_s4, W4, G, g, M, N, K, reps, true);

    printf("[registers] entry clamp (control) vs clamp + setmaxnreg donation\n");
    w[nw++] = run("clamp168", k_ws4_clamp168, W4, G, g, M, N, K, reps, true);
    w[nw++] = run("smr168/88,248", k_ws4_smr168, W4, G, g, M, N, K, reps, true);
    w[nw++] = run("clamp128", k_ws4_clamp128, W4, G, g, M, N, K, reps, true);
    w[nw++] = run("smr128/24,232", k_ws4_smr128, W4, G, g, M, N, K, reps, true);
    w[nw++] = run("smr128/88,168", k_ws4_smr128b, W4, G, g, M, N, K, reps, true);

    printf("[gate] FNV-1a over the whole C plane must equal the uniform arm's\n");
    int bad = 0;
    for (int i = 0; i < nw; i++) if (w[i].hash != u.hash) bad++;
    printf("  uniform %016llx   %d/%d ws arms differ  %s%s\n", (unsigned long long)u.hash, bad, nw,
           bad ? "FAIL" : "PASS", u.hash == zhash ? "  !! DEGENERATE: equals the zero plane" : "");
    printf("[summary] best ws vs uniform: ");
    double best = 1e30; int bi = 0;
    for (int i = 0; i < nw; i++) if (w[i].cpt < best && w[i].hash == u.hash) { best = w[i].cpt; bi = i; }
    printf("%.1f vs %.1f cyc/K-tile = %.3fx (arm %d)\n", best, u.cpt, u.cpt / best, bi);

    /* ---- GLU: two weight streams, two accumulator sets, ~2/3 of prefill GEMM FLOPs ---- */
    printf("[glu] cyc/K-tile is NOT comparable across N-tile widths — read ms / TF/s\n");
    const int G2 = 2 * (ABUF + 2 * 8192), G3 = 3 * (ABUF + 2 * 8192);
    const int H2 = 2 * (ABUF + 2 * 4096), H4 = 4 * (ABUF + 2 * 4096);
    const int V3 = MBAR_MAX + 3 * (ABUF + 2 * 4096), V4 = MBAR_MAX + 4 * (ABUF + 2 * 4096);
    const int V6 = MBAR_MAX + 6 * (ABUF + 2 * 4096);
    Res gu = run("glu u BN128 S2", k_glu_u128_s2, G2, G, g, M, N, K, reps, true, 128, 2);
    run("glu u BN128 S3", k_glu_u128_s3, G3, G, g, M, N, K, reps, true, 128, 2);
    Res gu64 = run("glu u BN64 S2", k_glu_u64_s2, H2, G, g, M, N, K, reps, true, 64, 2);
    run("glu u BN64 S4", k_glu_u64_s4, H4, G, g, M, N, K, reps, true, 64, 2);
    Res gw3 = run("glu ws BN64 S3", k_glu_ws64_s3, V3, G, g, M, N, K, reps, true, 64, 2);
    Res gw4 = run("glu ws BN64 S4", k_glu_ws64_s4, V4, G, g, M, N, K, reps, true, 64, 2);
    Res gw6 = run("glu ws BN64 S6", k_glu_ws64_s6, V6, G, g, M, N, K, reps, true, 64, 2);
    int gbad = (gu64.hash != gu.hash) + (gw3.hash != gu.hash) + (gw4.hash != gu.hash) +
               (gw6.hash != gu.hash);
    printf("  glu hash %016llx  %d/4 arms differ  %s%s\n", (unsigned long long)gu.hash, gbad,
           gbad ? "FAIL" : "PASS",
           gu.hash == u.hash ? "  !! DEGENERATE: identical to the plain-GEMM plane" : "");
    double gb = gw3.ms < gw4.ms ? gw3.ms : gw4.ms;
    if (gw6.ms < gb) gb = gw6.ms;
    printf("  glu best ws %.3f ms vs uniform BN128 %.3f ms = %.3fx, vs uniform BN64 %.3f ms = %.3fx\n",
           gb, gu.ms, gu.ms / gb, gu64.ms, gu64.ms / gb);
    (void)argc; (void)argv;
    return 0;
}
