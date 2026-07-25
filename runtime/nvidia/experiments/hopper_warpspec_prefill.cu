// Hopper (sm_90a) PRODUCER/CONSUMER WARP SPECIALIZATION + setmaxnreg on the
// PREFILL GEMM path.  Question: does the CUTLASS-style split (one producer
// warpgroup staging smem, N consumer warpgroups running wgmma, mbarrier
// handshake, setmaxnreg.dec/inc register rebalancing) beat the UNIFORM wgmma
// GEMM, and does it relieve the register pressure / spills that clamp the
// sm_90a prefill megakernel?
//
// Contract is identical to wgmma_bf16_probe.cu ("TN", both operands K-contig):
//   C[m,n] = sum_k A[m,k]*B[n,k],  A bf16 [M,K], B bf16 [N,K], C f32 [M,N].
// Descriptor / swizzle / epilogue recipe is REUSED VERBATIM from that file:
//   128-Byte swizzle, LBO=16 B, SBO=1024 B, m64n128k16, k16 substep = +32 B on
//   the descriptor start address, tile base rounded to 1024 B.
//
// Variants (all share tile shape BM=64*MS*CWG, BN=128, BK=64, NS stages):
//   uniform    128 thr, 1 warpgroup does both staging and wgmma  (== wgmma_bf16_probe.cu)
//   ws1_free   256 thr, 1 producer WG + 1 consumer WG, NO reg control (ptxas picks one count)
//   ws1_clamp  256 thr, same, __maxnreg__(ENTRY1)                  (uniform clamp, no donation)
//   ws1_smr    256 thr, same, __maxnreg__(ENTRY1) + dec(PROD1)/inc(CONS1)
//   ws2_*      384 thr, 1 producer WG + 2 consumer WGs, ENTRY2/PROD2/CONS2
// ws*_clamp is the CONTROL for ws*_smr: same entry register count -> same
// occupancy, so any delta is purely the producer->consumer register donation.
// Prior finding reused: the entry cap MUST be __maxnreg__(), NOT
// __launch_bounds__() -- ptxas silently drops the setmaxnreg effect with the latter.
//
// Pipeline: NS-deep smem ring, cp.async staged by the producer WG, completion
// published with `cp.async.mbarrier.arrive.noinc` on full[s]; consumers release
// a buffer with mbarrier.arrive on empty[s] after wgmma.wait_group<1> retires
// the group that read it.  Exactly one block-wide __syncthreads() (barrier init);
// after that the two warpgroups only ever meet on mbarriers.
//
// ============================ MEASURED (H100 NVL, 132 SM, CUDA 13.0) ============================
// All variants relL2 ~ 3.8e-6 at every shape -> correctness is not the issue.
//
// (1) setmaxnreg IS real on Hopper and ptxas DOES region-based allocation.
//     cuobjdump `REG:` reports the __maxnreg__ ENTRY value (what drives occupancy);
//     the consumer region really is compiled against the inc'd budget.
//     MS=2 (128 f32 accumulators/thread), entry=128, BOTH at 2 blk/SM:
//       ws1_clamp  REG:128  STACK:1024 B/thr  1320 B spill-st / 1576 B spill-ld
//                                             408 STL+LDL in SASS ->  33 TF/s
//       ws1_smr    REG:128  STACK:0           0 spills, 0 STL/LDL     -> 175 TF/s   (5.3x)
//     Same registers charged to the SM, same occupancy, spills gone. The mechanism works.
//
// (2) But there is nothing to donate, because the cp.async PRODUCER is not cheap.
//     ws1_smr TF/s at (512,4096,3840) as a function of the dec target (entry=128,
//     so cons = 256-prod; every point below has 2 blk/SM):
//       MS=1  prod=56 ->126   prod=72 ->165   prod=88 ->179   prod=96 ->138   prod=120 ->180
//       MS=2  prod=56 ->112   prod=72 ->131   prod=88 ->183   prod=104 ->40 (spills return)
//     Squeezing the producer below ~88 regs costs 20-45% throughput at ZERO spills:
//     it starves the staging loop's address generation and drops the number of
//     in-flight cp.async. The usable window is prod in [88,120] -> at most ~80
//     registers to donate, i.e. cons ~136..168. CUTLASS/DeepGEMM can dec to 24-40
//     because their producer is TMA (one thread, a descriptor and a barrier);
//     with cp.async the producer needs a real register file. That is the whole
//     difference, and it is structural, not a tuning miss.
//
// (3) Best-of-config TF/s, 100 iters + 10 warmup (config in parens MS/NS/variant),
//     sweeping MS in {1,2} x NS in {2,3,4} for every family:
//                        uniform          warpspec, no setmaxnreg      warpspec + setmaxnreg
//   (512, 4096,3840)   176.8 (1/3)       196.8 (2/4 ws1_free) +11.3%   187.9 (2/4 ws1_smr) +6.3%
//   (512,15360,3840)   219.5 (2/3)       217.8 (2/3 ws2_clamp) -0.8%   211.0 (2/3 ws1_smr) -3.9%
//   (200, 4096,3840)    83.9 (1/4)       103.0 (1/3 ws1_clamp)+22.8%   100.9 (1/3 ws1_smr)+20.3%
//   uniform reproduces wgmma_bf16_probe.cu exactly (177/163 at MS=1/NS=3, 168/219 at
//   MS=2/NS=3), so the comparison is against the already-validated 177 TF/s baseline.
//
// VERDICT: warp specialization itself is worth +11%/+23% at two of three prefill
// shapes and is a wash at the third (it wins where the epilogue and the ragged M tail
// cost the uniform kernel: the producer keeps streaming during the consumer's
// epilogue). setmaxnreg on top of it is NEVER the winner -- at every shape
// ws*_free / ws*_clamp >= ws*_smr. Its only real use is rescuing a kernel that is
// register-clamped AND spilling (case 1), and even then ws1_free at 154 regs /
// 1 blk/SM beats ws1_smr at 128 regs / 2 blk/SM (194.7 vs 183.3 at N=4096).
// So: do NOT put setmaxnreg into op_gemm.cuh's prefill path. Warp specialization
// per se is a shape-dependent ~10-20% and only pays if the selector picks it
// per shape; and to get the CUTLASS-style register donation to pay at all, the
// producer must first be moved from cp.async to TMA.
//
// Knobs: -DWS_MS (m64 slabs per consumer warpgroup, 1 or 2), -DWS_NS (ring depth).

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_bf16.h>

using bf16 = __nv_bfloat16;

#ifndef WS_MS
#define WS_MS 1
#endif
#ifndef WS_NS
#define WS_NS 3
#endif

static constexpr int MS = WS_MS;   // m64 wgmma slabs per consumer warpgroup
static constexpr int NS = WS_NS;   // smem ring depth
static constexpr int BN = 128;     // n128
static constexpr int BK = 64;      // staged K per stage -> 4 k16 wgmma substeps
static constexpr int WG = 128;     // threads per warpgroup
static constexpr int NACC = 64 * BN / WG;  // 64 f32 accumulators / thread / slab
static constexpr int KSUB = BK / 16;
static constexpr int KCHUNK = BK / 8;      // 16-byte chunks per row (= 128 B)
static constexpr int LBO = 16;
static constexpr int SBO = 1024;
static constexpr int SWZ_128B = 1;
static constexpr int BTILE = BN * BK;      // bf16 per B stage
static constexpr int MBAR_BYTES = 2 * NS * 8;

// entry / producer / consumer register counts per consumer-warpgroup count.
// Invariant (CTA-local pool is fixed by __maxnreg__ at entry):
//   PROD + CWG*CONS == (CWG+1)*ENTRY,  all multiples of 8, each in [24,256].
#ifndef WS_ENTRY1
#define WS_ENTRY1 128
#endif
#ifndef WS_PROD1
#define WS_PROD1 88
#endif
#ifndef WS_ENTRY2
#define WS_ENTRY2 168
#endif
#ifndef WS_PROD2
#define WS_PROD2 88
#endif
static constexpr int ENTRY1 = WS_ENTRY1, PROD1 = WS_PROD1, CONS1 = 2 * ENTRY1 - PROD1;
static constexpr int ENTRY2 = WS_ENTRY2, PROD2 = WS_PROD2, CONS2 = (3 * ENTRY2 - PROD2) / 2;
static_assert(PROD1 + CONS1 == 2 * ENTRY1 && CONS1 % 8 == 0 && CONS1 <= 256, "reg split 1");
static_assert(PROD2 + 2 * CONS2 == 3 * ENTRY2 && CONS2 % 8 == 0 && CONS2 <= 256, "reg split 2");

// ================= wgmma / cp.async / mbarrier primitives =================

__device__ __forceinline__ uint64_t desc_enc(uint64_t x) { return (x & 0x3FFFFull) >> 4; }
__device__ __forceinline__ uint64_t make_desc(const void* ptr, uint64_t lbo, uint64_t sbo) {
    uint64_t a = (uint64_t)__cvta_generic_to_shared(ptr);
    uint64_t d = 0;
    d |= desc_enc(a);
    d |= desc_enc(lbo) << 16;
    d |= desc_enc(sbo) << 32;
    d |= (uint64_t)SWZ_128B << 62;  // matrix base offset 0 (tile is 1024 B aligned)
    return d;
}
__device__ __forceinline__ int swz_off(int row, int c) { return row * BK + ((c ^ (row & 7)) * 8); }

__device__ __forceinline__ void cp16(void* smem, const void* gmem, int src_bytes) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem),
                 "r"(src_bytes));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::); }
__device__ __forceinline__ void wg_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::);
}
template <int N> __device__ __forceinline__ void wg_wait() {
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N));
}

__device__ __forceinline__ void mbar_init(uint64_t* b, int count) {
    unsigned a = (unsigned)__cvta_generic_to_shared(b);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(a), "r"(count));
}
__device__ __forceinline__ void mbar_arrive(uint64_t* b) {
    unsigned a = (unsigned)__cvta_generic_to_shared(b);
    asm volatile("{\n.reg .b64 st;\nmbarrier.arrive.shared::cta.b64 st, [%0];\n}\n" ::"r"(a));
}
// arrive-on-completion of ALL prior cp.async issued by this thread; .noinc means
// the initial expected-arrival count (= producer thread count) covers it.
__device__ __forceinline__ void cp_mbar_arrive(uint64_t* b) {
    unsigned a = (unsigned)__cvta_generic_to_shared(b);
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(a));
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, int parity) {
    unsigned a = (unsigned)__cvta_generic_to_shared(b);
    asm volatile("{\n"
                 ".reg .pred p;\n"
                 "WSW%=:\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
                 "@!p bra WSW%=;\n"
                 "}\n" ::"r"(a),
                 "r"(parity));
}

// one m64n128k16 .f32.bf16.bf16, both operands from smem (SS form)
__device__ __forceinline__ void wgmma_m64n128k16(float (&d)[NACC], uint64_t da, uint64_t db,
                                                 int scaleD) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %66, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,"
        "%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,"
        "%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, "
        "%64, %65, p, 1, 1, 0, 0;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
          "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]),
          "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]),
          "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
          "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]),
          "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]),
          "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]),
          "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
          "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]),
          "+f"(d[63])
        : "l"(da), "l"(db), "r"(scaleD));
}

// stage a [rows x BK] tile into 128B-swizzled smem; NTHR = staging thread count
template <int NTHR>
__device__ __forceinline__ void stage_swz(bf16* dst, const bf16* __restrict__ src, int tid,
                                          int rows, int row0, int kbase, int R, int K) {
    const int chunks = rows * KCHUNK;
    for (int L = tid; L < chunks; L += NTHR) {
        const int row = L / KCHUNK, c = L - row * KCHUNK;
        const int gr = row0 + row, gk = kbase + c * 8;
        int bytes = 0;
        const bf16* g = src;
        if (gr < R && gk < K) {
            g = src + (size_t)gr * K + gk;
            const int rem = K - gk;
            bytes = rem >= 8 ? 16 : rem * 2;
        }
        cp16(&dst[swz_off(row, c)], g, bytes);
    }
}

// round a shared-window pointer up to a 1024 B boundary (HW swizzle alignment)
__device__ __forceinline__ bf16* align1k(void* p) {
    unsigned off = (unsigned)__cvta_generic_to_shared(p) & 1023u;
    return (bf16*)((char*)p + ((1024u - off) & 1023u));
}

// ================= UNIFORM baseline (one warpgroup does everything) =================
static constexpr int U_BM = 64 * MS;
static constexpr int U_ATILE = U_BM * BK;
static constexpr int U_SMEM = NS * (U_ATILE + BTILE) * (int)sizeof(bf16) + 1024;

__global__ __launch_bounds__(WG) void k_uniform(float* __restrict__ C, const bf16* __restrict__ A,
                                                const bf16* __restrict__ B, int M, int N, int K) {
    extern __shared__ char plow_smem[];
    bf16* smem = align1k(plow_smem);
    bf16* As = smem;
    bf16* Bs = smem + NS * U_ATILE;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;

    const int mtiles = (M + U_BM - 1) / U_BM, ntiles = (N + BN - 1) / BN;
    const int total = mtiles * ntiles, ksteps = (K + BK - 1) / BK;

    for (int t = blockIdx.x; t < total; t += gridDim.x) {
        const int tm = (t / ntiles) * U_BM, tn = (t % ntiles) * BN;
        float acc[MS][NACC];
#pragma unroll
        for (int s = 0; s < NS - 1; s++) {
            if (s < ksteps) {
                stage_swz<WG>(As + s * U_ATILE, A, tid, U_BM, tm, s * BK, M, K);
                stage_swz<WG>(Bs + s * BTILE, B, tid, BN, tn, s * BK, N, K);
            }
            cp_commit();
        }
        for (int ks = 0; ks < ksteps; ks++) {
            const int cur = ks % NS;
            cp_wait<NS - 2>();
            __syncthreads();
            bf16* Ac = As + cur * U_ATILE;
            bf16* Bc = Bs + cur * BTILE;
            wg_fence();
#pragma unroll
            for (int sub = 0; sub < KSUB; sub++) {
                const int sd = (ks == 0 && sub == 0) ? 0 : 1;
                const uint64_t db = make_desc(Bc + sub * 16, LBO, SBO);
#pragma unroll
                for (int s = 0; s < MS; s++)
                    wgmma_m64n128k16(acc[s], make_desc(Ac + s * 64 * BK + sub * 16, LBO, SBO), db,
                                     sd);
            }
            wg_commit();
            wg_wait<1>();
            const int nxt = ks + NS - 1;
            if (nxt < ksteps) {
                const int nb = nxt % NS;
                stage_swz<WG>(As + nb * U_ATILE, A, tid, U_BM, tm, nxt * BK, M, K);
                stage_swz<WG>(Bs + nb * BTILE, B, tid, BN, tn, nxt * BK, N, K);
            }
            cp_commit();
        }
        wg_wait<0>();

        const int cbase = tn + (lane & 3) * 2;
#pragma unroll
        for (int s = 0; s < MS; s++) {
            const int r0 = tm + s * 64 + warp * 16 + (lane >> 2), r1 = r0 + 8;
#pragma unroll
            for (int j = 0; j < BN / 8; j++) {
                const int c = cbase + j * 8;
                if (r0 < M) {
                    if (c < N) C[(size_t)r0 * N + c] = acc[s][4 * j + 0];
                    if (c + 1 < N) C[(size_t)r0 * N + c + 1] = acc[s][4 * j + 1];
                }
                if (r1 < M) {
                    if (c < N) C[(size_t)r1 * N + c] = acc[s][4 * j + 2];
                    if (c + 1 < N) C[(size_t)r1 * N + c + 1] = acc[s][4 * j + 3];
                }
            }
        }
        __syncthreads();
    }
}

// ================= WARP-SPECIALIZED body =================
// CWG  = consumer warpgroups (block = 128*(CWG+1) threads, WG0 = producer)
// SMR  = emit setmaxnreg.dec/inc
template <int CWG> struct WsCfg {
    static constexpr int BM = 64 * MS * CWG;
    static constexpr int THREADS = 128 * (CWG + 1);
    static constexpr int ATILE = BM * BK;
    static constexpr int SMEM = MBAR_BYTES + NS * (ATILE + BTILE) * (int)sizeof(bf16) + 1024;
};

template <int CWG, bool SMR, int PREG, int CREG>
__device__ __forceinline__ void ws_body(float* __restrict__ C, const bf16* __restrict__ A,
                                        const bf16* __restrict__ B, int M, int N, int K) {
    constexpr int BM = WsCfg<CWG>::BM;
    constexpr int ATILE = WsCfg<CWG>::ATILE;

    extern __shared__ char plow_smem[];
    uint64_t* bfull = (uint64_t*)plow_smem;
    uint64_t* bempty = bfull + NS;
    bf16* smem = align1k(plow_smem + MBAR_BYTES);
    bf16* As = smem;
    bf16* Bs = smem + NS * ATILE;

    const int tid = threadIdx.x;
    if (tid < NS) {
        mbar_init(bfull + tid, WG);         // 128 producer threads publish a stage
        mbar_init(bempty + tid, WG * CWG);  // all consumer threads release a stage
    }
    __syncthreads();  // only block-wide sync; both branches are barrier-free afterwards

    const int mtiles = (M + BM - 1) / BM, ntiles = (N + BN - 1) / BN;
    const int total = mtiles * ntiles, ksteps = (K + BK - 1) / BK;

    if (tid < WG) {
        // ---------------- producer warpgroup ----------------
        if constexpr (SMR) asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(PREG));
        int st = 0;
        for (int t = blockIdx.x; t < total; t += gridDim.x) {
            const int tm = (t / ntiles) * BM, tn = (t % ntiles) * BN;
            for (int ks = 0; ks < ksteps; ks++, st++) {
                const int s = st % NS;
                if (st >= NS) mbar_wait(bempty + s, ((st / NS) + 1) & 1);
                stage_swz<WG>(As + s * ATILE, A, tid, BM, tm, ks * BK, M, K);
                stage_swz<WG>(Bs + s * BTILE, B, tid, BN, tn, ks * BK, N, K);
                cp_mbar_arrive(bfull + s);
            }
        }
    } else {
        // ---------------- consumer warpgroup(s) ----------------
        if constexpr (SMR) asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(CREG));
        const int ctid = tid - WG;
        const int cwg = ctid >> 7, lt = ctid & 127;
        const int warp = lt >> 5, lane = lt & 31;
        const int arow = cwg * 64 * MS;  // row offset of this consumer's slabs in the tile
        int st = 0;
        for (int t = blockIdx.x; t < total; t += gridDim.x) {
            const int tm = (t / ntiles) * BM + arow, tn = (t % ntiles) * BN;
            float acc[MS][NACC];
            int prev = -1;
            for (int ks = 0; ks < ksteps; ks++, st++) {
                const int s = st % NS;
                mbar_wait(bfull + s, (st / NS) & 1);
                bf16* Ac = As + s * ATILE + arow * BK;
                bf16* Bc = Bs + s * BTILE;
                wg_fence();
#pragma unroll
                for (int sub = 0; sub < KSUB; sub++) {
                    const int sd = (ks == 0 && sub == 0) ? 0 : 1;
                    const uint64_t db = make_desc(Bc + sub * 16, LBO, SBO);
#pragma unroll
                    for (int sl = 0; sl < MS; sl++)
                        wgmma_m64n128k16(acc[sl],
                                         make_desc(Ac + sl * 64 * BK + sub * 16, LBO, SBO), db, sd);
                }
                wg_commit();
                wg_wait<1>();  // group st-1 retired -> its buffer is free
                if (prev >= 0) mbar_arrive(bempty + prev);
                prev = s;
            }
            wg_wait<0>();
            if (prev >= 0) mbar_arrive(bempty + prev);

            const int cbase = tn + (lane & 3) * 2;
#pragma unroll
            for (int sl = 0; sl < MS; sl++) {
                const int r0 = tm + sl * 64 + warp * 16 + (lane >> 2), r1 = r0 + 8;
#pragma unroll
                for (int j = 0; j < BN / 8; j++) {
                    const int c = cbase + j * 8;
                    if (r0 < M) {
                        if (c < N) C[(size_t)r0 * N + c] = acc[sl][4 * j + 0];
                        if (c + 1 < N) C[(size_t)r0 * N + c + 1] = acc[sl][4 * j + 1];
                    }
                    if (r1 < M) {
                        if (c < N) C[(size_t)r1 * N + c] = acc[sl][4 * j + 2];
                        if (c + 1 < N) C[(size_t)r1 * N + c + 1] = acc[sl][4 * j + 3];
                    }
                }
            }
        }
    }
}

#define WS_FREE(name, CWG)                                                                       \
    __global__ void name(float* C, const bf16* A, const bf16* B, int M, int N, int K) {          \
        ws_body<CWG, false, 0, 0>(C, A, B, M, N, K);                                             \
    }
#define WS_CAP(name, CWG, ENTRY, SMR, PREG, CREG)                                                \
    __global__ void __maxnreg__(ENTRY)                                                            \
        name(float* C, const bf16* A, const bf16* B, int M, int N, int K) {                      \
        ws_body<CWG, SMR, PREG, CREG>(C, A, B, M, N, K);                                         \
    }

WS_FREE(k_ws1_free, 1)
WS_CAP(k_ws1_clamp, 1, ENTRY1, false, 0, 0)
WS_CAP(k_ws1_smr, 1, ENTRY1, true, PROD1, CONS1)
WS_FREE(k_ws2_free, 2)
WS_CAP(k_ws2_clamp, 2, ENTRY2, false, 0, 0)
WS_CAP(k_ws2_smr, 2, ENTRY2, true, PROD2, CONS2)

// ================= host: oracle, validation, benchmark =================
static uint32_t g_xs = 0x1234567u;
static float frand() {
    g_xs ^= g_xs << 13; g_xs ^= g_xs >> 17; g_xs ^= g_xs << 5;
    return ((g_xs >> 8) * (1.0f / 8388608.0f)) - 1.0f;
}
static float of_bf16(bf16 b) { return __bfloat162float(b); }

#define CK(x)                                                                                    \
    do {                                                                                         \
        cudaError_t e_ = (x);                                                                    \
        if (e_ != cudaSuccess) {                                                                 \
            printf("CUDA ERR %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));            \
            exit(1);                                                                             \
        }                                                                                        \
    } while (0)

static void oracle(std::vector<float>& C, const std::vector<bf16>& A, const std::vector<bf16>& B,
                   int M, int N, int K) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            const bf16* a = &A[(size_t)m * K];
            const bf16* b = &B[(size_t)n * K];
            float s = 0.f;
            for (int k = 0; k < K; k++) s += of_bf16(a[k]) * of_bf16(b[k]);
            C[(size_t)m * N + n] = s;
        }
}

using KFn = void (*)(float*, const bf16*, const bf16*, int, int, int);
struct Variant {
    const char* name;
    KFn fn;
    int threads;
    int bm;
    int smem;
    int occ;  // filled at init: blocks/SM
};

static Variant g_var[] = {
    {"uniform",   k_uniform,   WG,                  U_BM,            U_SMEM,            0},
    {"ws1_free",  k_ws1_free,  WsCfg<1>::THREADS,   WsCfg<1>::BM,    WsCfg<1>::SMEM,    0},
    {"ws1_clamp", k_ws1_clamp, WsCfg<1>::THREADS,   WsCfg<1>::BM,    WsCfg<1>::SMEM,    0},
    {"ws1_smr",   k_ws1_smr,   WsCfg<1>::THREADS,   WsCfg<1>::BM,    WsCfg<1>::SMEM,    0},
    {"ws2_free",  k_ws2_free,  WsCfg<2>::THREADS,   WsCfg<2>::BM,    WsCfg<2>::SMEM,    0},
    {"ws2_clamp", k_ws2_clamp, WsCfg<2>::THREADS,   WsCfg<2>::BM,    WsCfg<2>::SMEM,    0},
    {"ws2_smr",   k_ws2_smr,   WsCfg<2>::THREADS,   WsCfg<2>::BM,    WsCfg<2>::SMEM,    0},
};
static constexpr int NVAR = sizeof(g_var) / sizeof(g_var[0]);
static int g_sms = 132;

static int grid_for(const Variant& v, int M, int N) {
    const int total = ((M + v.bm - 1) / v.bm) * ((N + BN - 1) / BN);
    const int cap = g_sms * (v.occ > 0 ? v.occ : 1);
    return total < cap ? total : cap;
}

static void init_variants() {
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    g_sms = p.multiProcessorCount;
    printf("GPU %s  SMs=%d  cc %d.%d  regs/SM=%d  smem/SM=%zu KiB\n", p.name, g_sms, p.major,
           p.minor, p.regsPerMultiprocessor, p.sharedMemPerMultiprocessor / 1024);
    printf("tile: MS=%d NS=%d BN=%d BK=%d  (BM = 64*MS*consumerWGs)\n\n", MS, NS, BN, BK);
    printf("  reg split: CWG=1 entry=%d prod=%d cons=%d | CWG=2 entry=%d prod=%d cons=%d\n\n",
           ENTRY1, PROD1, CONS1, ENTRY2, PROD2, CONS2);
    printf("%-10s %7s %5s %9s %8s %10s %7s %11s\n", "variant", "threads", "BM", "smem KiB",
           "regs/thr", "spill B/thr", "blk/SM", "mathWGs/SM");
    for (int i = 0; i < NVAR; i++) {
        Variant& v = g_var[i];
        CK(cudaFuncSetAttribute((const void*)v.fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                v.smem));
        cudaFuncAttributes fa;
        CK(cudaFuncGetAttributes(&fa, (const void*)v.fn));
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v.occ, (const void*)v.fn, v.threads,
                                                         v.smem));
        if (v.occ < 1) v.occ = 1;
        const int mathwg = v.occ * (v.threads == WG ? 1 : (v.threads / WG - 1));
        printf("%-10s %7d %5d %9.1f %8d %10d %7d %11d\n", v.name, v.threads, v.bm,
               v.smem / 1024.0, fa.numRegs, (int)fa.localSizeBytes, v.occ, mathwg);
    }
    printf("\n");
}

struct Timing { double tf; };

static int run_shape(int M, int N, int K, bool bench) {
    std::vector<bf16> hA((size_t)M * K), hB((size_t)N * K);
    for (auto& x : hA) x = __float2bfloat16(frand());
    for (auto& x : hB) x = __float2bfloat16(frand());

    bf16 *dA, *dB;
    float* dC;
    CK(cudaMalloc(&dA, hA.size() * sizeof(bf16)));
    CK(cudaMalloc(&dB, hB.size() * sizeof(bf16)));
    CK(cudaMalloc(&dC, (size_t)M * N * sizeof(float)));
    CK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(bf16), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(bf16), cudaMemcpyHostToDevice));

    std::vector<float> ref((size_t)M * N), hC((size_t)M * N);
    oracle(ref, hA, hB, M, N, K);
    double den = 0;
    for (size_t i = 0; i < ref.size(); i++) den += (double)ref[i] * (double)ref[i];

    int fail = 0;
    printf("  (%4d,%5d,%5d):", M, N, K);
    double tf[NVAR];
    for (int i = 0; i < NVAR; i++) {
        Variant& v = g_var[i];
        const int grid = grid_for(v, M, N);
        CK(cudaMemset(dC, 0, (size_t)M * N * sizeof(float)));
        v.fn<<<grid, v.threads, v.smem>>>(dC, dA, dB, M, N, K);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float), cudaMemcpyDeviceToHost));
        double num = 0;
        for (size_t j = 0; j < hC.size(); j++) {
            double d = (double)hC[j] - (double)ref[j];
            num += d * d;
        }
        const double rel = sqrt(num / (den + 1e-30));
        const bool pass = rel < 3e-3;
        if (!pass) fail = 1;
        printf("  %s=%.2e %s", v.name, rel, pass ? "OK" : "FAIL");
        tf[i] = 0;

        if (bench && pass) {
            cudaEvent_t a, b;
            CK(cudaEventCreate(&a));
            CK(cudaEventCreate(&b));
            const int iters = 100, warm = 10;
            for (int it = 0; it < warm; it++) v.fn<<<grid, v.threads, v.smem>>>(dC, dA, dB, M, N, K);
            CK(cudaDeviceSynchronize());
            CK(cudaEventRecord(a));
            for (int it = 0; it < iters; it++) v.fn<<<grid, v.threads, v.smem>>>(dC, dA, dB, M, N, K);
            CK(cudaEventRecord(b));
            CK(cudaEventSynchronize(b));
            float ms;
            CK(cudaEventElapsedTime(&ms, a, b));
            tf[i] = 2.0 * M * N * K / (ms / 1e3 / iters) / 1e12;
            cudaEventDestroy(a);
            cudaEventDestroy(b);
        }
    }
    printf("\n");
    if (bench) {
        printf("    TF/s:");
        for (int i = 0; i < NVAR; i++)
            printf("  %s=%.1f%s", g_var[i].name, tf[i],
                   (i && tf[0] > 0) ? "" : "");
        printf("\n    vs uniform:");
        for (int i = 1; i < NVAR; i++)
            printf("  %s=%.3fx", g_var[i].name, tf[0] > 0 ? tf[i] / tf[0] : 0.0);
        printf("\n");
    }
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return fail;
}

int main(int argc, char** argv) {
    // argv[1] == "bench": skip the two big oracle checks (they dominate wall time
    // during a register-split sweep); the full run validates all three shapes.
    const bool quick = argc > 1 && argv[1][0] == 'b';
    printf("== Hopper warp specialization + setmaxnreg on the PREFILL wgmma GEMM (sm_90a) ==\n");
    init_variants();
    printf("CORRECTNESS (relL2 < 3e-3 vs f32 CPU oracle over the same bf16 inputs):\n");
    int fail = 0;
    fail |= run_shape(64, 128, 256, false);
    fail |= run_shape(256, 256, 320, false);
    if (!quick) {
        fail |= run_shape(512, 4096, 3840, false);
        fail |= run_shape(512, 15360, 3840, false);
        fail |= run_shape(200, 4096, 3840, false);
    }
    printf("RESULT: %s\n\n", fail ? "FAIL" : "PASS");
    if (fail) return 1;

    printf("BENCHMARK (100 iters + 10 warmup):\n");
    run_shape(512, 4096, 3840, true);
    run_shape(512, 15360, 3840, true);
    run_shape(200, 4096, 3840, true);
    return 0;
}
