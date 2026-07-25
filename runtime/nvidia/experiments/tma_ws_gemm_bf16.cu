// Hopper (sm_90a) FULL PIPELINE bf16 prefill GEMM:
//   TMA producer + producer/consumer warp specialization + 128 B swizzle +
//   large tiles + wgmma.  This is the flagship prefill kernel experiment.
//
// It closes the loop opened by runtime/nvidia/experiments/hopper_warpspec_prefill.cu,
// which concluded: "the precondition for the CUTLASS-style win is moving the
// producer from cp.async to TMA first; only then is there a cheap producer to
// donate from, and only then is setmaxnreg worth re-testing."  THIS FILE TESTS THAT.
//
// Contract (plow prefill linear, "TN", both operands K-contiguous) -- identical to
// wgmma_bf16_probe.cu / hopper_warpspec_prefill.cu:
//   C[m,n] = sum_k A[m,k]*B[n,k],  A bf16 [M,K], B bf16 [N,K], C f32 [M,N].
//
// ---------------------------------------------------------------------------
// WHY THE SWIZZLE RECIPE CARRIES OVER VERBATIM
// ---------------------------------------------------------------------------
// wgmma_bf16_probe.cu established, and validated, the SS-descriptor recipe for a
// 128-Byte-swizzled K-major smem tile:
//     LBO = 16 B, SBO = 1024 B, trans_a = trans_b = 0, desc bits[63:62] = 1,
//     physical element offset  row*64 + ((c ^ (row&7))*8)   (c = k/8, 16 B chunks),
//     one k16 wgmma substep = +32 B on the descriptor START ADDRESS only,
//     tile base MUST be 1024 B aligned (so matrix-base-offset stays 0 and the HW
//     address-bit swizzle [6:4]^[9:7] lines up).
// A TMA tensor map declared with CU_TENSOR_MAP_SWIZZLE_128B and an inner box dim
// of BK = 64 bf16 = 128 B produces EXACTLY that physical layout in smem -- the
// copy engine applies the XOR, so the store-side XOR is NOT written by hand here
// (that is the whole point: `swz_off()` disappears from the TMA path).  Every
// descriptor constant above is therefore reused unchanged, and the correctness
// gate below is what proves the two agree.
//
// ---------------------------------------------------------------------------
// STRUCTURE
// ---------------------------------------------------------------------------
// Block = 128*(CWG+1) threads.  Warpgroup 0 = producer, warpgroups 1..CWG = math.
//   producer: ONE elected thread (tid==0) per stage does
//       mbarrier.arrive.expect_tx(full[s], TX_BYTES)
//       cp.async.bulk.tensor.2d...mbarrier::complete_tx::bytes  x2  (A tile, B tile)
//     and waits on empty[s] (CWG arrivals) before reusing a buffer.  That is the
//     entire producer: no address generation, no per-thread loop, ~a dozen regs.
//   consumers: mbarrier.try_wait.parity(full[s]) -> wgmma m64n128k16 over
//     MS m-slabs x NSLB n-slabs -> wgmma.wait_group<1> -> one elected thread per
//     warpgroup arrives on empty[prev].
// Exactly one block-wide __syncthreads() (barrier init); after it the warpgroups
// meet only on mbarriers.
//
// TMA also removes the two hand-written bounds tests: out-of-bounds rows (ragged M)
// and the K tail are ZERO-FILLED by the copy engine, which is why the (200,...) and
// (64,128,16) shapes need no special casing at all.
//
// Tile shape knobs (compile time):
//   -DTWS_CWG   consumer warpgroups (1 or 2)
//   -DTWS_MS    m64 wgmma slabs per consumer warpgroup      -> BM = 64*MS*CWG
//   -DTWS_NSLB  n128 wgmma slabs                            -> BN = 128*NSLB
//   -DTWS_NS    smem ring depth
//   -DTWS_ENTRY __maxnreg__ at entry (drives occupancy)
//   -DTWS_PROD  setmaxnreg.dec target on the producer warpgroup
// Register invariant (CTA register pool is fixed by __maxnreg__ at entry):
//   PROD + CWG*CONS == (CWG+1)*ENTRY,  every value a multiple of 8, in [24,256].
// BN=256 is expressed as NSLB=2 x m64n128k16 rather than one m64n256k16: the two
// forms issue the same tensor-core work and the same 128 accumulators/thread, and
// NSLB=2 reuses the already-validated n128 descriptor + epilogue bit for bit.
//
// BUILD (executables MUST use the -gencode form; -arch=sm_90a alone does not
// forward the arch-accelerated feature set to ptxas, and -arch=native is sm_90):
//   env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin nvcc -std=c++17 \
//     -gencode arch=compute_90a,code=sm_90a -O3 -I runtime/common -I runtime/nvidia \
//     -include cstdint -Xcompiler -fopenmp \
//     runtime/nvidia/experiments/tma_ws_gemm_bf16.cu -lcuda -o BUILD/tma_ws
//   flock /tmp/plow_gpu.lock BUILD/tma_ws
// (-lcuda for cuTensorMapEncodeTiled, -fopenmp only for the CPU oracle.)
//
// ===========================================================================
// MEASURED (H100 NVL, 132 SM, CUDA 13.0, driver 570.133)
// ===========================================================================
// MEASUREMENT HYGIENE. This box was shared with an unrelated vLLM benchmark for most
// of the session, so every number below comes from the round-robin MIN timer (12
// blocks of 100 iters per variant, interleaved across variants, min kept) and is
// only quoted from runs where the two in-run BASELINES reproduce their independently
// validated values: uniform_cp = 175/219 TF/s (wgmma_bf16_probe.cu) and ws_cpasync
// = 197/218 TF/s (hopper_warpspec_prefill.cu). Rows that failed that gate are discarded,
// not averaged in.
//
// (1) CORRECTNESS -- every variant, every gate shape, relL2 vs an f32 CPU oracle over
//     the same bf16 inputs (threshold 3e-3):
//       (  64,   128,   16)  2.75e-08     ( 512,  4096, 3840)  3.79e-06
//       ( 200,  4096, 3840)  3.79e-06     ( 512, 15360, 3840)  3.78e-06
//     The TMA path needs NO bounds tests: ragged M and the K tail are zero-filled by
//     the copy engine, and the hardware's 128 B swizzle matches the hand-derived
//     wgmma descriptor recipe exactly -- (64,128,16) exercises K < BK, (200,...) M % BM.
//
// (2) THROUGHPUT, best config per family (TF/s; H100 bf16 dense peak = 989 TF/s):
//                            (512,4096,3840)      (512,15360,3840)     (200,4096,3840)
//   uniform cp.async wgmma       175.0  (17.7%)      219.3  (22.2%)       83.9  ( 8.5%)
//   warp-spec, cp.async prod     197.7  (20.0%)      217.8  (22.0%)      103.1  (10.4%)
//   warp-spec, TMA producer      391.0  (39.5%)      328.4  (33.2%)      223.7  (22.6%)
//   warp-spec, TMA + setmaxnreg  391.1  (39.5%)      332.5  (33.6%)      235.2  (23.8%)
//   TMA vs cp.async warp-spec     1.98x               1.53x               2.28x
//   TMA vs uniform                2.23x               1.52x               2.80x
//   Best (BM,BN,CWG,MS,NS,entry,dec):
//     (512, 4096): 128x128, CWG=1 MS=2, NS=6/7, entry 128, dec 32   -> 391
//     (512,15360): 128x128, CWG=1 MS=2, NS=3, entry 128, dec 32     -> 332  (2 blk/SM)
//     (200, 4096): 128x128, CWG=1 MS=2, NS=4..7, entry 128, dec 32  -> 235
//   One compromise default: NS=4  -> 381 / 263 / 231.
//   Tile-shape sweep (TMA+smr, best NS each): 128x128 CWG1 MS2 391/332/235;
//   128x128 CWG2 MS1 368/258/212; 256x128 277/328(tma)/130; 128x256 287/312/120;
//   64x256 343/230/202; 64x128 296/152/218.  128x256 and 256x128 only pay at
//   N=15360 -- at N=4096 they leave 64 output tiles for 132 SMs and starve the machine.
//
// (3) THE HEADLINE: setmaxnreg NOW PAYS, and the reason is exactly the one
//     hopper_warpspec_prefill.cu predicted.
//   Dec-target sweep, BM=128 BN=128 CWG=1 MS=2 (128 f32 accs/thread), entry=128,
//   so cons = 256 - dec.  NS=3 (smem 97 KiB -> 2 blk/SM is REACHABLE at 128 regs):
//     dec   cons   (512,4096)   (512,15360)   spill B/thr
//      24    232      364.8        332.5           0
//      32    224      364.4        332.8           0      <- flat
//      48    208      361.0        332.2           0
//      64    192      360.4        331.2           0
//      88    168      361.6        (contended)     0
//     128    128       32.8         31.7        1024      <- == the clamp control
//     no setmaxnreg (ws_tma, ptxas picks 156 regs -> 1 blk/SM):
//                      352.0        258.9           0
//   Two things changed versus the cp.async producer:
//     * The dec target is FLAT from 24 to 88. With cp.async, dropping the producer
//       below ~88 regs cost 20-45% at zero spills because its address generation
//       starved. A TMA producer is one elected thread issuing a descriptor and a
//       barrier: 24 registers is genuinely enough. The CUTLASS 24-40 window is real
//       here. The binding constraint moved to the CONSUMER (it needs ~168 regs for
//       128 accumulators + the wgmma pipeline; cons=152 and cons=128 both spill).
//     * Because the producer is that cheap, the entry clamp that buys 2 blk/SM
//       (entry=128 at 256 threads = 32768 regs) is affordable, and donation makes
//       the consumer whole. That is worth +3.6% at N=4096 and +28.4% at N=15360
//       over the un-clamped kernel -- the first time in this codebase that
//       setmaxnreg beats letting ptxas choose.
//   The ws_tma_clmp control is the proof it is the DONATION and not the clamp:
//   same entry=128, same occupancy, no setmaxnreg -> 1024 B/thread of spill,
//   172 STL[R1 + 240 LDL[R1 in SASS, 32 TF/s. The smr variant at the same 128-reg
//   entry has 0 spill and 0 STL/LDL. That is an 11x rescue (the cp.async file
//   measured 5.3x on the same mechanism).
//
// (4) REGISTERS / SPILLS (cudaFuncGetAttributes + `cuobjdump -sass` STL/LDL with [R1),
//     BM=128 BN=128 CWG=1 MS=2 NS=4, 256 threads:
//       variant        regs  spill B/thr  STL[R1  LDL[R1  blk/SM   UTMALDG  HGMMA  LDGSTS
//       uniform_cp      168        0         0       0      1         0       8      56
//       ws_cpasync      154        0         0       0      1         0       8      15
//       ws_tma          156        0         0       0      1        14       8       0
//       ws_tma_clmp     128     1032       172     240      1        14       8       0
//       ws_tma_smr      128        0         0       0      1(2@NS3) 14       8       0
//     CWG=2 MS=1 (64 accs/thread, 384 threads): uniform 140, ws_cpasync/ws_tma/clamp 90,
//     smr 128 -- no spills anywhere, and no setmaxnreg rescue to make (nothing spills).
//
// (5) SASS PROOF (cuobjdump -sass, function _Z9k_tma_smr...):
//       UTMALDG.2D [UR8], [UR6] ;                        x14  (TMA 2-D tile loads)
//       HGMMA.64x128x16.F32.BF16 R88, gdesc[UR4], R88 ;   x8  (warpgroup wgmma)
//       WARPGROUP.ARRIVE / WARPGROUP.DEPBAR.LE gsb0      (commit_group / wait_group)
//       SYNCS.PHASECHK.TRANS64.TRYWAIT                   (mbarrier try_wait.parity)
//       @P0 ELECT P2, URZ, PT                            (single-thread TMA issue)
//       LDGSTS: 0   -- cp.async is completely gone from the TMA kernels.
//     The cp.async kernels in the same cubin show LDGSTS 15/56 and UTMALDG 0.
//
// (6) ABI PROBE (`./tma_ws abi`, all PASS):
//       sizeof(CUtensorMap) = 128 B, alignof = 128 B (CUDA 13 header; the driver
//         documents a 64 B minimum, the type over-aligns).
//       cuTensorMapEncodeTiled = 45 ns/call on the host.
//       CUtensorMap in ORDINARY GLOBAL MEMORY, reached by pointer (not a
//         __grid_constant__ parameter): relL2 3.78e-06 PASS.
//       DEVICE-side `tensormap.replace.tile.global_address.global.b1024.b64`
//         + fence.proxy.tensormap::generic.release/acquire.gpu, re-pointing a live
//         descriptor at a different tensor: relL2 3.78e-06 PASS on sm_90a.
//     Both of the paths plow would need therefore exist. See the integration note
//     at the end of this file.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <ctime>

using bf16 = __nv_bfloat16;

// ------------------------------------------------------------------ config
#ifndef TWS_CWG
#define TWS_CWG 2
#endif
#ifndef TWS_MS
#define TWS_MS 1
#endif
#ifndef TWS_NSLB
#define TWS_NSLB 1
#endif
#ifndef TWS_NS
#define TWS_NS 4
#endif
#ifndef TWS_ENTRY
#define TWS_ENTRY 128
#endif
#ifndef TWS_PROD
#define TWS_PROD 32
#endif

static constexpr int CWG  = TWS_CWG;
static constexpr int MS   = TWS_MS;
static constexpr int NSLB = TWS_NSLB;
static constexpr int NS   = TWS_NS;

static constexpr int WG   = 128;                 // threads per warpgroup
static constexpr int BK   = 64;                  // 128 B inner box == 128 B swizzle atom
static constexpr int BN   = 128 * NSLB;
static constexpr int BM   = 64 * MS * CWG;
static constexpr int NACC = 64;                  // f32 accs / thread / (m-slab, n-slab)
static constexpr int KSUB = BK / 16;             // wgmma k16 substeps per stage
static constexpr int LBO  = 16;                  // leading   byte offset (128 B swizzle)
static constexpr int SBO  = 1024;                // stride    byte offset (8 rows x 128 B)
static constexpr int SWZ_128B = 1;               // descriptor bits[63:62]

static constexpr int ATILE = BM * BK;            // bf16 per A stage
static constexpr int BTILE = BN * BK;            // bf16 per B stage
static constexpr int TXB   = (ATILE + BTILE) * (int)sizeof(bf16);   // TMA bytes / stage
static constexpr int THREADS = WG * (CWG + 1);

// register split
static constexpr int ENTRY = TWS_ENTRY;
static constexpr int PROD  = TWS_PROD;
static constexpr int CONS  = ((CWG + 1) * ENTRY - PROD) / CWG;
static_assert(PROD + CWG * CONS == (CWG + 1) * ENTRY, "reg split must be exact");
static_assert(CONS % 8 == 0 && CONS <= 256 && PROD % 8 == 0 && PROD >= 24, "reg split range");

// uniform (single-warpgroup) cp.async baseline tile: one WG can only carry
// MS*NSLB*64 accumulators, so it keeps BM = 64*MS and the same BN.
static constexpr int U_BM    = 64 * MS;
static constexpr int U_ATILE = U_BM * BK;

// ---------------------------------------------------- wgmma / mbarrier / TMA
__device__ __forceinline__ uint32_t su32(const void* p) {
    return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ uint64_t desc_enc(uint64_t x) { return (x & 0x3FFFFull) >> 4; }
__device__ __forceinline__ uint64_t make_desc(const void* ptr, uint64_t lbo, uint64_t sbo) {
    uint64_t d = 0;
    d |= desc_enc((uint64_t)__cvta_generic_to_shared(ptr));
    d |= desc_enc(lbo) << 16;
    d |= desc_enc(sbo) << 32;
    d |= (uint64_t)SWZ_128B << 62;   // matrix base offset 0 (tile base is 1024 B aligned)
    return d;
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::); }
template <int N> __device__ __forceinline__ void wg_wait() {
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N));
}

__device__ __forceinline__ void mbar_init(uint64_t* b, int cnt) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(su32(b)), "r"(cnt) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* b) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(su32(b)) : "memory");
}
// arrive + declare the transaction byte count this phase must also collect
__device__ __forceinline__ void mbar_expect(uint64_t* b, int bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(su32(b)),
                 "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, int parity) {
    asm volatile("{\n.reg .pred p;\nTW%=:\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
                 "@!p bra TW%=;\n}\n" ::"r"(su32(b)), "r"(parity) : "memory");
}
// TMA 2-D tile load, single destination CTA. c0 = inner (K) coord, c1 = row coord.
__device__ __forceinline__ void tma2d(uint32_t dst, const CUtensorMap* map, int c0, int c1,
                                      uint32_t bar) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
                 " [%0], [%1, {%2, %3}], [%4];\n" ::"r"(dst),
                 "l"(map), "r"(c0), "r"(c1), "r"(bar) : "memory");
}

// cp.async path (baselines only)
__device__ __forceinline__ void cp16(void* smem, const void* gmem, int src_bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(su32(smem)), "l"(gmem),
                 "r"(src_bytes));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void cp_mbar_arrive(uint64_t* b) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(su32(b)) : "memory");
}
__device__ __forceinline__ int swz_off(int row, int c) { return row * BK + ((c ^ (row & 7)) * 8); }
template <int NTHR>
__device__ __forceinline__ void stage_swz(bf16* dst, const bf16* __restrict__ src, int tid,
                                          int rows, int row0, int kbase, int R, int K) {
    const int chunks = rows * (BK / 8);
    for (int L = tid; L < chunks; L += NTHR) {
        const int row = L / (BK / 8), c = L - row * (BK / 8);
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

__device__ __forceinline__ bf16* align1k(void* p) {
    unsigned off = su32(p) & 1023u;
    return (bf16*)((char*)p + ((1024u - off) & 1023u));
}

// one m64n128k16 .f32.bf16.bf16, both operands from smem (SS form).
// scaleD 0 -> D = A*B (seeds accs, no zeroing pass); 1 -> D = A*B + D.
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

// ------------------------------------------- shared consumer math + epilogue
// acc[MS][NSLB][64].  Issues the MS x NSLB wgmma grid for one BK stage.
template <int MSx, int NSLBx>
__device__ __forceinline__ void wgmma_stage(float (&acc)[MSx][NSLBx][NACC], const bf16* Ac,
                                            const bf16* Bc, int seed) {
    wg_fence();
#pragma unroll
    for (int sub = 0; sub < KSUB; sub++) {
        const int sd = (seed && sub == 0) ? 0 : 1;
#pragma unroll
        for (int nb = 0; nb < NSLBx; nb++) {
            const uint64_t db = make_desc(Bc + nb * 128 * BK + sub * 16, LBO, SBO);
#pragma unroll
            for (int sl = 0; sl < MSx; sl++)
                wgmma_m64n128k16(acc[sl][nb], make_desc(Ac + sl * 64 * BK + sub * 16, LBO, SBO), db,
                                 sd);
        }
    }
    wg_commit();
}

// warpgroup m64n128 f32 accumulator -> (row, col):
//   warp w owns rows [16w, 16w+16); for n-block j: acc[4j+0..3] map to
//   (16w+lane/4, 8j+(lane%4)*2+{0,1}) and the same columns 8 rows down.
template <int MSx, int NSLBx>
__device__ __forceinline__ void epilogue(float* __restrict__ C, float (&acc)[MSx][NSLBx][NACC],
                                         int tm, int tn, int warp, int lane, int M, int N) {
    const int cbase = tn + (lane & 3) * 2;
#pragma unroll
    for (int sl = 0; sl < MSx; sl++) {
        const int r0 = tm + sl * 64 + warp * 16 + (lane >> 2), r1 = r0 + 8;
#pragma unroll
        for (int nb = 0; nb < NSLBx; nb++) {
#pragma unroll
            for (int j = 0; j < 16; j++) {
                const int c = cbase + nb * 128 + j * 8;
                if (r0 < M) {
                    if (c < N) C[(size_t)r0 * N + c] = acc[sl][nb][4 * j + 0];
                    if (c + 1 < N) C[(size_t)r0 * N + c + 1] = acc[sl][nb][4 * j + 1];
                }
                if (r1 < M) {
                    if (c < N) C[(size_t)r1 * N + c] = acc[sl][nb][4 * j + 2];
                    if (c + 1 < N) C[(size_t)r1 * N + c + 1] = acc[sl][nb][4 * j + 3];
                }
            }
        }
    }
}

// ===================== baseline 1: UNIFORM cp.async wgmma =====================
// == wgmma_bf16_probe.cu (the validated 177 / 219 TF/s kernel), NSLB-generalised.
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
        float acc[MS][NSLB][NACC];
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
            wgmma_stage<MS, NSLB>(acc, As + cur * U_ATILE, Bs + cur * BTILE, ks == 0);
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
        epilogue<MS, NSLB>(C, acc, tm, tn, warp, lane, M, N);
        __syncthreads();
    }
}

// ============ baseline 2: warp specialization with a cp.async producer ============
// == hopper_warpspec_prefill.cu ws*_free (the 196.8 TF/s class).
static constexpr int MBAR_BYTES = 2 * NS * 8;
static constexpr int WS_SMEM = MBAR_BYTES + NS * (ATILE + BTILE) * (int)sizeof(bf16) + 1024;

__global__ void k_ws_cp(float* __restrict__ C, const bf16* __restrict__ A,
                        const bf16* __restrict__ B, int M, int N, int K) {
    extern __shared__ char plow_smem[];
    uint64_t* bfull = (uint64_t*)plow_smem;
    uint64_t* bempty = bfull + NS;
    bf16* smem = align1k(plow_smem + MBAR_BYTES);
    bf16* As = smem;
    bf16* Bs = smem + NS * ATILE;
    const int tid = threadIdx.x;
    if (tid < NS) {
        mbar_init(bfull + tid, WG);
        mbar_init(bempty + tid, WG * CWG);
    }
    __syncthreads();

    const int mtiles = (M + BM - 1) / BM, ntiles = (N + BN - 1) / BN;
    const int total = mtiles * ntiles, ksteps = (K + BK - 1) / BK;

    if (tid < WG) {
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
        const int ctid = tid - WG, cwg = ctid >> 7, lt = ctid & 127;
        const int warp = lt >> 5, lane = lt & 31, arow = cwg * 64 * MS;
        int st = 0;
        for (int t = blockIdx.x; t < total; t += gridDim.x) {
            const int tm = (t / ntiles) * BM + arow, tn = (t % ntiles) * BN;
            float acc[MS][NSLB][NACC];
            int prev = -1;
            for (int ks = 0; ks < ksteps; ks++, st++) {
                const int s = st % NS;
                mbar_wait(bfull + s, (st / NS) & 1);
                wgmma_stage<MS, NSLB>(acc, As + s * ATILE + arow * BK, Bs + s * BTILE, ks == 0);
                wg_wait<1>();
                if (prev >= 0) mbar_arrive(bempty + prev);
                prev = s;
            }
            wg_wait<0>();
            if (prev >= 0) mbar_arrive(bempty + prev);
            epilogue<MS, NSLB>(C, acc, tm, tn, warp, lane, M, N);
        }
    }
}

// ======================== THE TMA WARP-SPECIALIZED KERNEL ========================
template <bool SMR, int PREG, int CREG, bool TMAP_GMEM = false>
__device__ __forceinline__ void tma_body(float* __restrict__ C, const CUtensorMap* mapA,
                                         const CUtensorMap* mapB, int M, int N, int K) {
    extern __shared__ char plow_smem[];
    uint64_t* bfull = (uint64_t*)plow_smem;
    uint64_t* bempty = bfull + NS;
    bf16* smem = align1k(plow_smem + MBAR_BYTES);   // 1024 B aligned: HW swizzle + desc agree
    bf16* As = smem;                                 // [NS][BM][64] 128 B-swizzled by TMA
    bf16* Bs = smem + NS * ATILE;                    // [NS][BN][64]

    const int tid = threadIdx.x;
    if (tid < NS) {
        mbar_init(bfull + tid, 1);      // one elected producer thread arrives (+ tx bytes)
        mbar_init(bempty + tid, CWG);   // one elected thread per consumer warpgroup
    }
    __syncthreads();   // only block-wide sync in the kernel

    const int mtiles = (M + BM - 1) / BM, ntiles = (N + BN - 1) / BN;
    const int total = mtiles * ntiles, ksteps = (K + BK - 1) / BK;

    if (tid < WG) {
        // ------------------------------- producer warpgroup -------------------------------
        if constexpr (SMR) asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(PREG));
        if constexpr (TMAP_GMEM) {
            // The tensor map lives in ordinary global memory (not a __grid_constant__
            // param) and may have been written by a previous DEVICE-side
            // tensormap.replace, which is a write in the tensormap proxy. Reading it
            // through cp.async.bulk.tensor needs an explicit acquire in that proxy.
            if (tid == 0) {
                asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;\n" ::"l"(mapA)
                             : "memory");
                asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;\n" ::"l"(mapB)
                             : "memory");
            }
        }
        if (tid == 0) {
            int st = 0;
            for (int t = blockIdx.x; t < total; t += gridDim.x) {
                const int tm = (t / ntiles) * BM, tn = (t % ntiles) * BN;
                for (int ks = 0; ks < ksteps; ks++, st++) {
                    const int s = st % NS;
                    if (st >= NS) mbar_wait(bempty + s, ((st / NS) + 1) & 1);
                    const uint32_t bar = su32(bfull + s);
                    mbar_expect(bfull + s, TXB);
                    tma2d(su32(As + s * ATILE), mapA, ks * BK, tm, bar);
                    tma2d(su32(Bs + s * BTILE), mapB, ks * BK, tn, bar);
                }
            }
        }
    } else {
        // ------------------------------ consumer warpgroups ------------------------------
        if constexpr (SMR) asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(CREG));
        const int ctid = tid - WG, cwg = ctid >> 7, lt = ctid & 127;
        const int warp = lt >> 5, lane = lt & 31, arow = cwg * 64 * MS;
        int st = 0;
        for (int t = blockIdx.x; t < total; t += gridDim.x) {
            const int tm = (t / ntiles) * BM + arow, tn = (t % ntiles) * BN;
            float acc[MS][NSLB][NACC];
            int prev = -1;
            for (int ks = 0; ks < ksteps; ks++, st++) {
                const int s = st % NS;
                mbar_wait(bfull + s, (st / NS) & 1);
                wgmma_stage<MS, NSLB>(acc, As + s * ATILE + arow * BK, Bs + s * BTILE, ks == 0);
                wg_wait<1>();   // group st-1 retired -> its buffer is free
                if (prev >= 0 && lt == 0) mbar_arrive(bempty + prev);
                prev = s;
            }
            wg_wait<0>();
            if (prev >= 0 && lt == 0) mbar_arrive(bempty + prev);
            epilogue<MS, NSLB>(C, acc, tm, tn, warp, lane, M, N);
        }
    }
}

__global__ void k_tma_free(float* C, const __grid_constant__ CUtensorMap mapA,
                           const __grid_constant__ CUtensorMap mapB, int M, int N, int K) {
    tma_body<false, 0, 0>(C, &mapA, &mapB, M, N, K);
}
__global__ void __maxnreg__(ENTRY)
    k_tma_clamp(float* C, const __grid_constant__ CUtensorMap mapA,
                const __grid_constant__ CUtensorMap mapB, int M, int N, int K) {
    tma_body<false, 0, 0>(C, &mapA, &mapB, M, N, K);
}
__global__ void __maxnreg__(ENTRY)
    k_tma_smr(float* C, const __grid_constant__ CUtensorMap mapA,
              const __grid_constant__ CUtensorMap mapB, int M, int N, int K) {
    tma_body<true, PROD, CONS>(C, &mapA, &mapB, M, N, K);
}

// ---------------- ABI probe: tensor maps that are NOT kernel parameters ----------------
// plow's interpreter is ONE persistent megakernel whose parameters are fixed for the life
// of the model, so a __grid_constant__ CUtensorMap parameter per GEMM is structurally
// impossible: the descriptor has to be reachable from a device pointer the packet carries.
// These two kernels test exactly that path.
//   k_tma_gmem  : identical math, but the maps are read from ordinary global memory.
//   k_tmap_patch: DEVICE-side  tensormap.replace.tile.global_address  -- re-points an
//                 existing 128 B descriptor at another tensor with the same shape, which
//                 is what a per-layer / per-expert weight rebind would need.
__global__ void k_tma_gmem(float* C, const CUtensorMap* mapA, const CUtensorMap* mapB, int M,
                           int N, int K) {
    tma_body<false, 0, 0, true>(C, mapA, mapB, M, N, K);
}
__global__ void k_tmap_patch(CUtensorMap* map, void* new_addr) {
    if (threadIdx.x == 0) {
        asm volatile("tensormap.replace.tile.global_address.global.b1024.b64 [%0], %1;\n" ::"l"(map),
                     "l"(new_addr) : "memory");
        asm volatile("fence.proxy.tensormap::generic.release.gpu;\n" ::: "memory");
    }
}

// ============================== host harness ==============================
#define CK(x)                                                                                      \
    do {                                                                                           \
        cudaError_t e_ = (x);                                                                      \
        if (e_ != cudaSuccess) {                                                                   \
            printf("CUDA ERR %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));             \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)
#define CKD(x)                                                                                     \
    do {                                                                                           \
        CUresult e_ = (x);                                                                         \
        if (e_ != CUDA_SUCCESS) {                                                                  \
            const char* s_;                                                                        \
            cuGetErrorString(e_, &s_);                                                             \
            printf("CU ERR %s:%d %s\n", __FILE__, __LINE__, s_);                                   \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

// 2-D tensor map over a [rows, K] bf16 k-contiguous tensor, box {BK, boxRows},
// 128 B swizzle. Out-of-bounds (ragged M, K tail) is zero-filled by the copy engine.
static void make_map(CUtensorMap* m, void* base, int rows, int K, int boxRows) {
    uint64_t gd[2] = {(uint64_t)K, (uint64_t)rows};
    uint64_t gs[1] = {(uint64_t)K * 2};
    uint32_t bd[2] = {(uint32_t)BK, (uint32_t)boxRows};
    uint32_t es[2] = {1, 1};
    memset(m, 0, sizeof(*m));
    CKD(cuTensorMapEncodeTiled(m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, base, gd, gs, bd, es,
                               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                               CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

static uint32_t g_xs = 0x1234567u;
static float frand() {
    g_xs ^= g_xs << 13; g_xs ^= g_xs >> 17; g_xs ^= g_xs << 5;
    return ((g_xs >> 8) * (1.0f / 8388608.0f)) - 1.0f;
}

// f32 CPU oracle over the SAME bf16 values the kernel reads
static void oracle(std::vector<float>& C, const std::vector<float>& A, const std::vector<float>& B,
                   int M, int N, int K) {
#pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            const float* a = &A[(size_t)m * K];
            const float* b = &B[(size_t)n * K];
            float s = 0.f;
            for (int k = 0; k < K; k++) s += a[k] * b[k];
            C[(size_t)m * N + n] = s;
        }
}

enum Kind { K_UNI, K_WSCP, K_TMA };
struct Variant {
    const char* name;
    Kind kind;
    const void* fn;
    int threads, bm, smem, occ, regs, spill;
};
static Variant g_var[] = {
    {"uniform_cp",  K_UNI,  (const void*)k_uniform,   WG,      U_BM, U_SMEM,  0, 0, 0},
    {"ws_cpasync",  K_WSCP, (const void*)k_ws_cp,     THREADS, BM,   WS_SMEM, 0, 0, 0},
    {"ws_tma",      K_TMA,  (const void*)k_tma_free,  THREADS, BM,   WS_SMEM, 0, 0, 0},
    {"ws_tma_clmp", K_TMA,  (const void*)k_tma_clamp, THREADS, BM,   WS_SMEM, 0, 0, 0},
    {"ws_tma_smr",  K_TMA,  (const void*)k_tma_smr,   THREADS, BM,   WS_SMEM, 0, 0, 0},
};
static constexpr int NVAR = (int)(sizeof(g_var) / sizeof(g_var[0]));
static int g_sms = 132;

static int grid_for(const Variant& v, int M, int N) {
    const int total = ((M + v.bm - 1) / v.bm) * ((N + BN - 1) / BN);
    const int cap = g_sms * (v.occ > 0 ? v.occ : 1);
    return total < cap ? total : cap;
}

static void launch(const Variant& v, int grid, float* C, const bf16* A, const bf16* B,
                   const CUtensorMap& mA, const CUtensorMap& mB, int M, int N, int K) {
    switch (v.kind) {
        case K_UNI:  k_uniform<<<grid, v.threads, v.smem>>>(C, A, B, M, N, K); break;
        case K_WSCP: k_ws_cp<<<grid, v.threads, v.smem>>>(C, A, B, M, N, K); break;
        case K_TMA:
            if (v.fn == (const void*)k_tma_free)
                k_tma_free<<<grid, v.threads, v.smem>>>(C, mA, mB, M, N, K);
            else if (v.fn == (const void*)k_tma_clamp)
                k_tma_clamp<<<grid, v.threads, v.smem>>>(C, mA, mB, M, N, K);
            else
                k_tma_smr<<<grid, v.threads, v.smem>>>(C, mA, mB, M, N, K);
            break;
    }
}

static void init_variants() {
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    g_sms = p.multiProcessorCount;
    printf("GPU %s  SMs=%d  cc %d.%d  regs/SM=%d  smem/SM=%zu KiB\n", p.name, g_sms, p.major,
           p.minor, p.regsPerMultiprocessor, p.sharedMemPerMultiprocessor / 1024);
    printf("tile: BM=%d BN=%d BK=%d  CWG=%d MS=%d NSLB=%d NS=%d  accs/thr=%d  TMA tx/stage=%d B\n",
           BM, BN, BK, CWG, MS, NSLB, NS, MS * NSLB * NACC, TXB);
    printf("regs: entry=%d prod=%d cons=%d   (uniform baseline tile BM=%d)\n\n", ENTRY, PROD, CONS,
           U_BM);
    printf("%-12s %7s %5s %9s %8s %10s %7s %10s\n", "variant", "threads", "BM", "smem KiB",
           "regs/thr", "spill B/thr", "blk/SM", "mathWG/SM");
    for (int i = 0; i < NVAR; i++) {
        Variant& v = g_var[i];
        CK(cudaFuncSetAttribute(v.fn, cudaFuncAttributeMaxDynamicSharedMemorySize, v.smem));
        cudaFuncAttributes fa;
        CK(cudaFuncGetAttributes(&fa, v.fn));
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v.occ, v.fn, v.threads, v.smem));
        if (v.occ < 1) v.occ = 1;
        v.regs = fa.numRegs;
        v.spill = (int)fa.localSizeBytes;
        const int mathwg = v.occ * (v.threads == WG ? 1 : (v.threads / WG - 1));
        printf("%-12s %7d %5d %9.1f %8d %10d %7d %10d\n", v.name, v.threads, v.bm, v.smem / 1024.0,
               v.regs, v.spill, v.occ, mathwg);
    }
    printf("\n");
}

static double g_tf[NVAR];

static int run_shape(int M, int N, int K, bool bench, bool check) {
    std::vector<bf16> hA((size_t)M * K), hB((size_t)N * K);
    std::vector<float> fA(hA.size()), fB(hB.size());
    for (size_t i = 0; i < hA.size(); i++) { hA[i] = __float2bfloat16(frand()); fA[i] = __bfloat162float(hA[i]); }
    for (size_t i = 0; i < hB.size(); i++) { hB[i] = __float2bfloat16(frand()); fB[i] = __bfloat162float(hB[i]); }

    bf16 *dA, *dB;
    float* dC;
    CK(cudaMalloc(&dA, hA.size() * sizeof(bf16)));
    CK(cudaMalloc(&dB, hB.size() * sizeof(bf16)));
    CK(cudaMalloc(&dC, (size_t)M * N * sizeof(float)));
    CK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(bf16), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(bf16), cudaMemcpyHostToDevice));

    CUtensorMap mA, mB;
    make_map(&mA, dA, M, K, BM);
    make_map(&mB, dB, N, K, BN);

    std::vector<float> ref, hC((size_t)M * N);
    double den = 0;
    if (check) {
        ref.resize((size_t)M * N);
        oracle(ref, fA, fB, M, N, K);
        for (size_t i = 0; i < ref.size(); i++) den += (double)ref[i] * (double)ref[i];
    }

    int fail = 0;
    if (check) printf("  (%4d,%6d,%5d):", M, N, K);
    for (int i = 0; i < NVAR; i++) {
        Variant& v = g_var[i];
        const int grid = grid_for(v, M, N);
        g_tf[i] = 0;
        if (check) {
            CK(cudaMemset(dC, 0, (size_t)M * N * sizeof(float)));
            launch(v, grid, dC, dA, dB, mA, mB, M, N, K);
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
        }
    }
    // ROUND-ROBIN timing.  This box is shared with unrelated GPU jobs, so a single
    // 100-iter block is not a measurement of this kernel -- it is a measurement of
    // whatever else happened to be resident.  Each ROUND times every variant once
    // (100 iters + 10 warmup on the first round) and we keep the MIN time per
    // variant across rounds: adjacent-in-time blocks see the same interference,
    // and the min lands in a quiet window if one exists.
    if (bench && !fail) {
        const int iters = 100, warm = 10;
        const char* rs = getenv("TWS_ROUNDS");
        const int rounds = rs ? atoi(rs) : 12;
        double bestms[NVAR];
        for (int i = 0; i < NVAR; i++) bestms[i] = 1e30;
        cudaEvent_t a, b;
        CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        for (int i = 0; i < NVAR; i++) {
            const int grid = grid_for(g_var[i], M, N);
            for (int it = 0; it < warm; it++) launch(g_var[i], grid, dC, dA, dB, mA, mB, M, N, K);
        }
        CK(cudaDeviceSynchronize());
        for (int r = 0; r < rounds; r++)
            for (int i = 0; i < NVAR; i++) {
                const int grid = grid_for(g_var[i], M, N);
                CK(cudaEventRecord(a));
                for (int it = 0; it < iters; it++)
                    launch(g_var[i], grid, dC, dA, dB, mA, mB, M, N, K);
                CK(cudaEventRecord(b));
                CK(cudaEventSynchronize(b));
                float ms;
                CK(cudaEventElapsedTime(&ms, a, b));
                if (ms / iters < bestms[i]) bestms[i] = ms / iters;
            }
        cudaEventDestroy(a); cudaEventDestroy(b);
        for (int i = 0; i < NVAR; i++) g_tf[i] = 2.0 * M * N * K / (bestms[i] / 1e3) / 1e12;
    }
    if (check) printf("\n");
    if (bench) {
        printf("  BENCH (%4d,%6d,%5d):", M, N, K);
        for (int i = 0; i < NVAR; i++)
            printf("  %s=%.1f(g%d)", g_var[i].name, g_tf[i], grid_for(g_var[i], M, N));
        printf("   [peak%% ");
        for (int i = 0; i < NVAR; i++) printf(" %.1f", 100.0 * g_tf[i] / 989.0);
        printf("]\n");
    }
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return fail;
}

// ---- ABI probe: can plow reach TMA without a __grid_constant__ kernel parameter? ----
// (1) host cost of cuTensorMapEncodeTiled, (2) map read from global memory, (3) map
// re-pointed at a different tensor by a DEVICE-side tensormap.replace.
static void abi_probe() {
    const int M = 512, N = 4096, K = 3840;
    printf("== ABI PROBE: tensor maps outside the kernel-parameter path ==\n");
    std::vector<bf16> hA((size_t)M * K), hB((size_t)N * K), hA2((size_t)M * K);
    std::vector<float> fA(hA.size()), fB(hB.size()), fA2(hA.size());
    for (size_t i = 0; i < hA.size(); i++) {
        hA[i] = __float2bfloat16(frand()); fA[i] = __bfloat162float(hA[i]);
        hA2[i] = __float2bfloat16(frand()); fA2[i] = __bfloat162float(hA2[i]);
    }
    for (size_t i = 0; i < hB.size(); i++) { hB[i] = __float2bfloat16(frand()); fB[i] = __bfloat162float(hB[i]); }
    bf16 *dA, *dA2, *dB;
    float* dC;
    CK(cudaMalloc(&dA, hA.size() * 2)); CK(cudaMalloc(&dA2, hA.size() * 2));
    CK(cudaMalloc(&dB, hB.size() * 2)); CK(cudaMalloc(&dC, (size_t)M * N * 4));
    CK(cudaMemcpy(dA, hA.data(), hA.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dA2, hA2.data(), hA2.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB.data(), hB.size() * 2, cudaMemcpyHostToDevice));

    // (1) host encode cost
    CUtensorMap mA, mB;
    cudaEvent_t junk;
    CK(cudaEventCreate(&junk));
    const int NENC = 2000;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < NENC; i++) make_map(&mA, dA, M, K, BM);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    const double enc_ns = ((t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec)) / NENC;
    make_map(&mB, dB, N, K, BN);
    printf("  sizeof(CUtensorMap)=%zu B, alignof=%zu B, cuTensorMapEncodeTiled = %.0f ns/call\n",
           sizeof(CUtensorMap), alignof(CUtensorMap), enc_ns);

    // (2) maps in ordinary global memory, reached through a pointer
    CUtensorMap *gA, *gB;
    CK(cudaMalloc(&gA, sizeof(CUtensorMap))); CK(cudaMalloc(&gB, sizeof(CUtensorMap)));
    CK(cudaMemcpy(gA, &mA, sizeof(mA), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(gB, &mB, sizeof(mB), cudaMemcpyHostToDevice));
    CK(cudaFuncSetAttribute((const void*)k_tma_gmem, cudaFuncAttributeMaxDynamicSharedMemorySize,
                            WS_SMEM));
    int occ = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, (const void*)k_tma_gmem, THREADS,
                                                     WS_SMEM));
    const int total = ((M + BM - 1) / BM) * ((N + BN - 1) / BN);
    const int grid = std::min(total, g_sms * std::max(occ, 1));

    std::vector<float> ref((size_t)M * N), hC((size_t)M * N);
    auto check = [&](const std::vector<float>& Asrc, const char* what) {
        oracle(ref, Asrc, fB, M, N, K);
        double num = 0, den = 0;
        CK(cudaMemcpy(hC.data(), dC, hC.size() * 4, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < ref.size(); i++) {
            double d = (double)hC[i] - (double)ref[i];
            num += d * d; den += (double)ref[i] * (double)ref[i];
        }
        const double rel = sqrt(num / (den + 1e-30));
        printf("  %-58s relL2=%.2e %s\n", what, rel, rel < 3e-3 ? "PASS" : "FAIL");
        return rel < 3e-3;
    };

    CK(cudaMemset(dC, 0, (size_t)M * N * 4));
    k_tma_gmem<<<grid, THREADS, WS_SMEM>>>(dC, gA, gB, M, N, K);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("  map-in-global-memory: LAUNCH FAIL %s\n", cudaGetErrorString(e)); cudaGetLastError(); }
    else check(fA, "(2) CUtensorMap in GLOBAL MEMORY, passed by pointer");

    // (3) device-side tensormap.replace: re-point the SAME descriptor at dA2
    k_tmap_patch<<<1, 32>>>(gA, dA2);
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        printf("  (3) device tensormap.replace.global_address: FAIL %s\n", cudaGetErrorString(e));
        cudaGetLastError();
    } else {
        CK(cudaMemset(dC, 0, (size_t)M * N * 4));
        k_tma_gmem<<<grid, THREADS, WS_SMEM>>>(dC, gA, gB, M, N, K);
        e = cudaDeviceSynchronize();
        if (e != cudaSuccess) { printf("  (3) run after replace: FAIL %s\n", cudaGetErrorString(e)); cudaGetLastError(); }
        else check(fA2, "(3) after DEVICE-side tensormap.replace.global_address -> A2");
    }
    cudaEventDestroy(junk);
    cudaFree(dA); cudaFree(dA2); cudaFree(dB); cudaFree(dC); cudaFree(gA); cudaFree(gB);
    printf("\n");
}

int main(int argc, char** argv) {
    const bool quick = argc > 1 && argv[1][0] == 'q';   // sweep mode: small oracles only
    CKD(cuInit(0));
    printf("== TMA + warp-spec + 128B swizzle + wgmma bf16 prefill GEMM (sm_90a) ==\n");
    init_variants();

    printf("CORRECTNESS (relL2 < 3e-3 vs f32 CPU oracle over the same bf16 inputs):\n");
    int fail = 0;
    fail |= run_shape(64, 128, 16, false, true);
    fail |= run_shape(200, 4096, 3840, false, true);
    if (!quick) {
        fail |= run_shape(512, 4096, 3840, false, true);
        fail |= run_shape(512, 15360, 3840, false, true);
    }
    printf("RESULT: %s\n\n", fail ? "FAIL" : "PASS");
    if (fail) return 1;
    if (argc > 1 && argv[1][0] == 'a') { abi_probe(); return 0; }

    printf("BENCHMARK (100 iters + 10 warmup, TF/s; H100 bf16 dense peak 989 TF/s):\n");
    run_shape(512, 4096, 3840, true, false);
    run_shape(512, 15360, 3840, true, false);
    run_shape(200, 4096, 3840, true, false);
    return 0;
}

/* ===========================================================================
 * INTEGRATION NOTE: what plow's packet/ABI would have to carry for TMA
 * ===========================================================================
 * TODAY. PlowDevInst is a fixed 64-byte record: op, blocks, fj[3], t[8] (uint16
 * tensor HANDLES), i[8]. The interpreter resolves a handle with
 *   #define TEN(k) (PLOW_T(k) == PLOW_TENSOR_NONE ? nullptr : T[PLOW_T(k)])
 * i.e. a raw device pointer out of ctx.tensors[]. Nothing in the packet, and
 * nothing in PlowDevCtx, is a descriptor. A CUtensorMap is 128 B and is bound to
 * (global address, rank, globalDim[], globalStride[], boxDim[], elementStride[],
 * dtype, interleave, swizzle, L2 promotion, OOB fill) -- so it is bound to the
 * TILE SHAPE as well as the tensor. Two GEMMs with the same weight but different
 * BM/BN need two descriptors.
 *
 * WHAT WOULD HAVE TO CHANGE. The megakernel's parameters are fixed for the life of
 * the model, so a __grid_constant__ CUtensorMap parameter per GEMM is structurally
 * impossible. The probe above shows the alternative works: keep a device-resident
 * CUtensorMap[] (128 B each, naturally aligned) and reach it by pointer. Concretely:
 *   1. PlowDevCtx gains one field:  const CUtensorMap* tmaps;   (+ uint32_t n_tmaps)
 *      This is additive at the end of the struct, so it does not move `trace` and
 *      the crates/packet/tests/dev_abi.rs layout lock only sees the size grow --
 *      the same discipline the cross-GPU fields already used.
 *   2. The GEMM packet needs TWO descriptor ids (A and B). There are no spare t[]
 *      slots semantically (they are tensor handles, not descriptor ids), but the
 *      prefill GEMM does not use all 8 i[] words -- two uint32 i[] slots, or one
 *      i[] word packing two uint16 ids, is enough. No struct growth required.
 *   3. The host builds the table. Descriptor identity is (tensor, tile shape), and
 *      the tile shape is chosen by the SELECTOR (plans/arch-gemm-tuning-system.md),
 *      which is exactly where a descriptor id can be minted.
 *
 * HOW MANY DESCRIPTORS A GEMMA-4 PREFILL NEEDS.
 *   Gemma-4 E4B: 42 layers, hidden 2560, inter 10240, q 8x256, kv 2x256, vocab 262144.
 *   WEIGHT (B) side, one descriptor per (matrix, tile shape):
 *     q, k, v, o, gate, up, down = 7 per layer  x 42 = 294, plus lm_head = 295.
 *     These are STATIC: address and shape are fixed at model load, so they are built
 *     ONCE (295 x 45 ns = 13 us, one time) and never touched again. 295 x 128 B =
 *     38 KB of device memory. If the selector picks more than one tile shape across
 *     the network, multiply by the number of distinct shapes actually used (2-3
 *     realistically -> <= ~900 descriptors, ~115 KB). Trivial either way.
 *   ACTIVATION (A) side: the descriptor depends on (arena slot address, M = token
 *     count, K). Arena slots are reused, so there are only a handful of distinct
 *     (address, K) pairs -- but M changes per prefill call. That is O(10)
 *     re-encodes per prefill, ~450 ns of host time, which is noise next to a
 *     prefill. Alternative: encode A with a FIXED max-M globalDim and let TMA
 *     zero-fill the rows past the real M (this file already relies on that OOB
 *     zero-fill for the ragged-M case), which makes the A descriptors static too.
 *   Gemma-4 26B-A4B MoE is the one place it gets big: 128 experts x 3 matrices x
 *     30 layers = 11520 descriptors (1.4 MB) if built per expert. Two ways out:
 *     (a) one 3-D tensor map over [expert][N][K] with the expert as a TMA
 *     coordinate -- one descriptor for the whole layer; or (b) device-side
 *     tensormap.replace per group, which probe (3) proves works on sm_90a.
 *
 * IS DEVICE-SIDE CONSTRUCTION/REPLACEMENT A VIABLE ALTERNATIVE ON sm_90a?
 *   Replacement: YES, measured. `tensormap.replace.tile.global_address` on a
 *   descriptor in global memory, plus fence.proxy.tensormap::generic.release.gpu on
 *   the writer and .acquire.gpu on the reader, retargeted a live descriptor and the
 *   next GEMM produced the correct result (relL2 3.78e-06). The same instruction
 *   family can replace global_dim / global_stride / box_dim, so a persistent kernel
 *   CAN own a small descriptor pool and rebind it per op.
 *   Construction from scratch: NO -- there is no device-side cuTensorMapEncodeTiled.
 *   `tensormap.replace` only edits FIELDS of an existing, host-encoded descriptor,
 *   so the host must still mint at least one template per (dtype, rank, swizzle,
 *   box shape). That template is cheap (45 ns) and there are only a few.
 *   The real cost of the device-side path is not the replace, it is the fencing:
 *   the tensormap proxy release/acquire pair is a GPU-scope fence, and a persistent
 *   interpreter that rebinds per packet would pay it per packet. Given that the
 *   weight descriptors are STATIC and number ~300, the honest recommendation is:
 *   build them host-side at load, carry an id in the packet, and keep
 *   tensormap.replace in reserve for the MoE grouped case where the descriptor
 *   count would otherwise be 11k.
 *
 * REMAINING BLOCKER THAT IS NOT ABOUT DESCRIPTORS. This kernel is a standalone
 * grid-stride kernel with its own smem budget (97-193 KiB) and its own register
 * split. plow's interpreter is ONE persistent kernel whose smem arena and register
 * budget are shared by every op in the switch, and whose block shape is fixed
 * (see the co-residency invariant in dev_isa.h). Dropping a 2-warpgroup, 128 KiB,
 * setmaxnreg-partitioned GEMM into that switch changes the megakernel's occupancy
 * for EVERY op. That is a bigger design question than the ABI, and it is the same
 * one plans/arch-gemm-tuning-system.md is already circling.
 */
