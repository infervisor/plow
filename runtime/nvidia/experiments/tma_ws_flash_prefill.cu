/* tma_ws_flash_prefill.cu — Hopper (sm_90a) WGMMA flash-attention PREFILL, extended with a
 * TMA producer, a multi-stage K/V pipeline, producer/consumer warp specialization + setmaxnreg,
 * and a causal tile skip.
 *
 * STARTS FROM the validated probe runtime/nvidia/experiments/wgmma_flash_prefill_probe.cu
 * (2.82x over an oracle-checked FA-2 mma.sync baseline). Its two load-bearing findings are
 * PRESERVED VERBATIM here:
 *   (a) SOFTMAX STAYS IN-FRAGMENT. A lane owns exactly two rows in BOTH accumulators, so
 *       scale/causal-mask/rowmax/exp/rowsum/rescale happen in registers with the row reduction
 *       as a quad butterfly (__shfl_xor 1 then 2). Routing S through smem measured 0.785x.
 *   (b) V IS MN-MAJOR (trans-b=1), consumed in its natural [BKV][HD] layout. Transposing V in
 *       smem cost 46% of runtime. LBO steps along K, SBO along N.
 *
 * WHAT CHANGES vs the probe — and it is exactly ONE thing, the smem LAYOUT of the wgmma operands:
 *   The probe stores Q/K/V CORE-MATRIX PACKED with descriptor swizzle mode 0 (LBO=128,
 *   SBO=16*W). TMA cannot produce that layout. TMA can produce the canonical Hopper 128-BYTE
 *   SWIZZLE layout, which the HARDWARE applies (CU_TENSOR_MAP_SWIZZLE_128B in the tensor map) —
 *   so there is NO store-side XOR here, and the wgmma descriptors must switch to swizzle mode 1
 *   with the swizzled stride convention. Concretely, for a [rows][64] bf16 sub-tile (row pitch
 *   128 B, swizzle atom = 8 rows = 1024 B):
 *     K-MAJOR  operand (Q, K): LBO = 16 B, SBO = 1024 B
 *     MN-MAJOR operand (V)   : LBO = 16 B, SBO = 1024 B   <-- MEASURED, SAME as K-major
 *   That last line is the new result and it is NOT what the no-swizzle case would predict: the
 *   probe had to SWAP the roles for MN-major (LBO=16*W, SBO=128). Under a swizzle mode the smem
 *   layout is self-describing to the hardware, so the strides stop depending on majorness and
 *   only trans-b does. Selftest 2 below runs BOTH variants and the swapped one fails at
 *   relL2 1.007e+00 while LBO=16/SBO=1024 lands at 6.5e-08 (the smem window is over-allocated
 *   and zero-padded so the losing variant reads defined memory instead of faulting).
 *   Ps (the GEMM1 A operand) is written from registers by the consumer, so it KEEPS the probe's
 *   swizzle-0 core-matrix-packed layout untouched.
 *
 * Because 128B swizzle fixes the innermost box at 64 bf16 = 128 B, every [rows][HD] tile is
 * stored as HD/64 independent 1024-B-aligned [rows][64] sub-tiles, one TMA issue each. For V
 * that is a free win: sub-tile nt IS the n64 output tile of GEMM1.
 *
 * TENSOR MAPS are 3-D — {HD, seq, heads} — NOT 2-D over a flattened [heads*seq][HD]. The third
 * dim makes the per-head row range a real tensor boundary, so TMA's out-of-bounds ZERO FILL
 * handles the ragged sequence tail instead of silently reading the next head's rows. That also
 * deletes the probe's explicit tail zero-fill code.
 *
 * VARIANTS
 *   probe      128 thr, single-buffered cp.async, swizzle-0 packed   (verbatim reference)
 *   uni        128 thr, TMA producer = thread 0, NS-stage K/V ring
 *   ws         256 thr, producer warpgroup (thread 0 issues TMA) + consumer warpgroup, no regctl
 *   wsr        256 thr, same + __maxnreg__(128) & setmaxnreg.dec(PROD)/inc(256-PROD)
 *   any of the above with CSKIP=1 adds the causal tile skip (KV tiles entirely above the
 *   diagonal are never staged and never computed) — reported separately because it changes the
 *   flop count, so both raw ms and flop-adjusted TF/s are printed.
 *
 * BUILD (the explicit -gencode is REQUIRED; -arch=sm_90a emits compute_90 PTX and every wgmma
 * is rejected, and -arch=native resolves to sm_90):
 *   env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin nvcc -std=c++17 \
 *       -gencode arch=compute_90a,code=sm_90a -O3 \
 *       -I runtime/common -I runtime/nvidia -include cstdint \
 *       runtime/nvidia/experiments/tma_ws_flash_prefill.cu -lcuda -o <bin>
 *
 * ==================== MEASURED (H100 NVL, 132 SM, CUDA 13.0) ==================================
 * CORRECTNESS: 12/12 cases x 4 variants PASS vs the probe's f32 CPU oracle, worst relL2
 * 1.686e-03 (gate 3e-3, probe was 1.770e-03). The causal-skip variants are BIT-comparable to
 * the non-skip ones at every case, confirming the skipped tiles were fully masked anyway.
 *
 * SPEED (best-of-3 ms, ratio vs the probe measured IN THE SAME RUN; zero spills everywhere,
 * SASS census below). CAVEAT: another job was resident on this GPU during the sweep, so
 * ABSOLUTE TF/s is depressed vs the probe's published 43.6 (the probe re-measured 21.8 here).
 * Ratios are same-run and therefore sound; absolute numbers are not.
 *
 *   shape                                     probe   tma uni  tma ws  tma wsr  best
 *                                                       NS=2    NS=2   NS=2 d24
 *   A 512x4096 nh8   (64 blk, UNDER-FILLED)   1.00x    2.65x   1.84x    2.80x   3.09x (wsr d32)
 *   B 512x4096 nh32  (256 blk, fills)         1.00x    2.76x   1.93x    2.67x   2.78x (uni+cskip)
 *   C 512x16384 nh8  (64 blk, UNDER-FILLED)   1.00x    2.62x   1.69x    2.88x   2.88x (wsr d24)
 *   D 4096x4096 nh8  (512 blk, fills)         1.00x    2.83x   2.16x    2.84x   5.01x (wsr d32+cskip)
 *
 * (1) TMA + double buffering is the whole win: 2.6-2.8x over the single-buffered cp.async probe
 *     at every shape. LDGSTS 19 -> 0, UTMALDG.3D 0 -> 10.
 * (2) DEPTH BEYOND NS=2 NEVER PAYS. NS=2 is 90 KiB -> 2 blk/SM; NS=3 is 122 KiB -> 1 blk/SM and
 *     loses 30-40% at both machine-filling shapes (B 2.76x -> 1.74x, D 2.83x -> 1.80x) and is a
 *     wash at the under-filled ones. The headroom was "any pipelining at all", not depth.
 * (3) WARP SPECIALIZATION ALONE IS A REGRESSION here (1.7-2.2x vs uniform's 2.6-2.8x) and the
 *     cause is a ptxas codegen pathology, not the split: ptxas emits
 *       "(C7520) wgmma.mma_async instructions are serialized due to program dependence on
 *        compiler-inserted WG.AR in divergent path"
 *     for k_tma_ws, and the SASS shows WARPGROUP.DEPBAR.LE 2 -> 8 (every wgmma group forced to
 *     drain). This is the OPPOSITE sign of hopper_warpspec_prefill.cu's GEMM result (+11-23%),
 *     and the difference is that flash's consumer body is one divergent branch of the CTA.
 * (4) setmaxnreg DOES NOT DONATE ANYTHING USEFUL -- it REPAIRS (3). k_tma_wsr differs from
 *     k_tma_ws only by __maxnreg__(128) + dec/inc, and that alone puts WARPGROUP.DEPBAR.LE back
 *     to 2 and recovers 1.84x -> 2.80x. But it only ties/slightly beats plain UNIFORM, and it
 *     LOSES to uniform at the machine-filling shape B (2.67x vs 2.76x). The producer here is one
 *     thread issuing 4 UTMALDG, so hopper_warpspec_prefill.cu's precondition ("move the producer
 *     to TMA, then re-test setmaxnreg") is now satisfied -- and the answer is still NO: with the
 *     consumer already at 125-128 regs and ZERO spills, there is nothing for the extra ~100
 *     donated registers to do. dec target barely matters (d24/d32/d40 within 8%).
 * (5) CAUSAL TILE SKIP is worth exactly what the flop count says and only where seq_q ~ seq_kv.
 *     Shape D (full prefill) computes 50.8% of the dense flops -> 0.4564 -> 0.2731 ms = 1.67x on
 *     top of TMA, i.e. 4.72x total for uni NS=2. Shapes A/B (chunked prefill, seq_q=512 into
 *     seq_kv=4096) are at 94.5% of dense -> under 3%, and C at 98.6% -> noise. The flop-adjusted
 *     TF/s FALLS when skipping (D uni NS=2: 150.6 dense-equivalent -> 127.8 real) because the
 *     surviving work is load-imbalanced: q-tile 0 runs 1 KV tile and q-tile 63 runs 64. Wall
 *     time is the honest comparator; the flop-adjusted number says the machine got LESS
 *     efficient per flop while finishing sooner.
 *
 * BEST CONFIG: NS=2 everywhere. Uniform (128 thr) when the grid fills the machine; wsr NS=2
 * d24/d32 when it does not (A: 3.09x, C: 2.88x). Plus the causal tile skip, always -- it is free
 * when it does not apply and 1.67x when it does.
 *
 * SASS PROOF (cuobjdump -sass, per-kernel; whole binary has 379 UTMALDG, 209 HGMMA, 0 STL/LDL):
 *   kernel                    UTMALDG.3D  HGMMA  WG.ARRIVE  WG.DEPBAR  USETMAXREG  LDGSTS  spill
 *   k_probe_flash                      0      5          9          2           0      19      0
 *   k_tma_uni  NS=2                   10      8          7          2           0       0      0
 *   k_tma_ws   NS=2                   14      8          8          8           0       0      0   <- (3)
 *   k_tma_wsr  NS=2 d32               14      8          7          2           2       0      0
 *   USETMAXREG.TRY_ALLOC.CTAPOOL / .DEALLOC.CTAPOOL = setmaxnreg.inc / .dec really landed.
 *   regs/thread: probe 123, uni 125, ws 124-127, wsr 128 (= the __maxnreg__ entry cap).
 *   REAL spills (STL/LDL against [R1]) are ZERO in every kernel in the binary.
 *
 * ==================== KV DESCRIPTOR / PACKET-ABI VERDICT ======================================
 * VERDICT: TMA IS VIABLE FOR KV in plow as it is shipped today, with NO per-step descriptor
 * rebuild. The intuition that "KV's base address and length change per step" does not survive
 * contact with plow's actual cache layout.
 *   - LENGTH: cuTensorMapEncodeTiled bounds the tensor, it does not bound the live sequence.
 *     Encode globalDim[seq] = kv_stride (the RING CAPACITY / compiled context), not seq_kv. The
 *     live length is already enforced IN-FRAGMENT by the existing `kvp < seq_kv -> NEG_BIG`
 *     mask, so a growing seq_kv costs zero descriptor churn. TMA's OOB zero-fill is then only
 *     doing the ragged-tile tail, which is what it does in this file.
 *   - BASE ADDRESS: op_attention.cuh's cache is [n_batch][n_kv_head][kv_stride][D] -- ONE
 *     contiguous per-layer allocation with uniform strides (memory/prefix.rs: "exactly ONE
 *     contiguous run per (b, kv_head)", explicitly NOT a page table). So batch slot, kv head and
 *     kv row are all TMA COORDINATES of a single rank-4 map {D, kv_stride, n_kv_head, n_batch},
 *     box {64, BKV, 1, 1}. The base pointer is fixed for the life of the allocation. Even
 *     d_flash_prefill_mux's per-request `K + kvoff` becomes coordinate 3, not a new descriptor.
 *   - RING WRAP: kv_mask = kv_stride-1 => kv_stride is a power of two => BKV | kv_stride, so a
 *     KV tile never straddles the wrap. One box per tile, always.
 *   - VMM-backed KV (memory/vmm.rs) is fine: the descriptor holds a VA, and VMM grows PHYSICAL
 *     pages behind a reserved VA. The map survives growth.
 * WHAT THE PACKET/ABI MUST CARRY: a tensor map is a 128-byte opaque, 64-byte-aligned blob and
 * cuTensorMapEncodeTiled is HOST-ONLY, so it cannot live in a 28-byte PlowFlashBody. It does not
 * need to: add a tensormap BufKind and pass it through the EXISTING handle path -- one more
 * t[] slot resolved as TEN(n), pointing at a 128-B device buffer built once at KV-allocation
 * time (per layer, per K/V: ~40x2x128 B = 10 KiB for a 40-layer model, not per slot and not per
 * step). Zero packet-format change beyond the slot; the interpreter already resolves handles
 * device-side with no host round-trip.
 * WHERE IT WOULD BREAK (and does not, today): a PAGED KV pool with a per-request block table
 * (memory/kv.rs has the allocator but prefix.rs says the shipped path is contiguous). There the
 * base changes per TILE, and the only device-side fix is tensormap.replace +
 * fence.proxy.tensormap::generic in the mainloop -- a proxy fence per tile, which would give
 * back most of what TMA just won. If plow ever moves to paged KV, TMA prefill must be gated off
 * or restricted to Q/weights.
 */
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x)                                                                                      \
    do {                                                                                           \
        cudaError_t e_ = (x);                                                                      \
        if (e_ != cudaSuccess) {                                                                   \
            printf("CUDA ERROR %s @ %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__);         \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)
#define CKD(x)                                                                                     \
    do {                                                                                           \
        CUresult e_ = (x);                                                                         \
        if (e_ != CUDA_SUCCESS) {                                                                  \
            const char* s_;                                                                        \
            cuGetErrorString(e_, &s_);                                                             \
            printf("CU ERROR %s @ %s:%d\n", s_, __FILE__, __LINE__);                               \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

typedef __nv_bfloat16 bf16;
#define NEG_BIG (-3.0e38f)

/* ============================ wgmma descriptors ==============================================
 * 64-bit shared-memory matrix descriptor:
 *   bits [ 0,14) start address >> 4
 *   bits [16,30) leading-dim byte offset (LBO) >> 4     <- K-direction core-matrix step
 *   bits [32,46) stride-dim byte offset (SBO) >> 4      <- MN-direction core-matrix step
 *   bits [49,52) matrix base offset (0: every tile base is 1024 B aligned)
 *   bits [62,64) swizzle mode: 0 none, 1 = 128 B, 2 = 64 B, 3 = 32 B
 * With swizzle != 0 the descriptor carries the LOGICAL (unswizzled) offsets and the hardware
 * applies the XOR itself — which is exactly why a TMA-written tile needs no store-side fixup. */
__device__ __forceinline__ uint64_t desc_enc(uint64_t x) { return (x & 0x3FFFFull) >> 4; }

__device__ __forceinline__ uint64_t make_desc(const void* ptr, uint64_t lbo, uint64_t sbo,
                                              int swz) {
    uint64_t a = (uint64_t)__cvta_generic_to_shared(ptr);
    uint64_t d = 0;
    d |= desc_enc(a);
    d |= desc_enc(lbo) << 16;
    d |= desc_enc(sbo) << 32;
    d |= (uint64_t)swz << 62;
    return d;
}
/* 128B-swizzled [rows][64] sub-tile, K-MAJOR (contraction along the contiguous 64) */
__device__ __forceinline__ uint64_t desc_kmajor_sw128(const void* p) {
    return make_desc(p, 16, 1024, 1);
}
/* 128B-swizzled [K][64] sub-tile, MN-MAJOR (contraction along rows) — variant selected by
 * selftest 2. VAR 0: LBO=1024 (K = 8 rows), SBO=16 (N = 8 elems).  VAR 1: swapped. */
template <int VAR> __device__ __forceinline__ uint64_t desc_mnmajor_sw128(const void* p) {
    return VAR == 0 ? make_desc(p, 1024, 16, 1) : make_desc(p, 16, 1024, 1);
}

/* --- the probe's swizzle-0 core-matrix-packed layout, kept for Ps and for the probe kernel --- */
__device__ __forceinline__ uint64_t make_smem_desc0(const void* ptr, uint32_t row_w_elems) {
    return make_desc(ptr, 128u, 16u * row_w_elems, 0);
}
__device__ __forceinline__ int cm_off(int r, int c, int W) {
    return (r >> 3) * (W * 8) + (c >> 3) * 64 + (r & 7) * 8 + (c & 7);
}
#define CM_KSLICE(ptr, ks) ((ptr) + (ks) * 128)

/* 128B swizzle, in ELEMENTS, for a [rows][64] bf16 sub-tile. Used ONLY by the selftests (to
 * reproduce in software what TMA does in hardware) and by the layout-equality check. */
__host__ __device__ __forceinline__ int sw128_off(int row, int col64) {
    const int chunk = col64 >> 3; /* 16-byte chunk index within the 128-byte row */
    return row * 64 + ((chunk ^ (row & 7)) << 3) + (col64 & 7);
}

/* ACCUMULATOR FRAGMENT LAYOUT (f32, m64nNk16) — unchanged from the probe. A thread owns exactly
 * TWO rows (rA = 16w+lane/4, rB = rA+8) in BOTH accumulators, which is what keeps the online
 * softmax rescale of O a two-value register operation. */
#define ACC_ROW_A(w, lane) ((w) * 16 + ((lane) >> 2))
#define ACC_ROW_B(w, lane) ((w) * 16 + ((lane) >> 2) + 8)
#define ACC_COL(nb, lane, e) ((nb) * 8 + 2 * ((lane) & 3) + (e))

__device__ __forceinline__ void wgmma_fence() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void wgmma_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void wgmma_wait0() {
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}
__device__ __forceinline__ void async_proxy_fence() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_m64n64k16(float (&d)[32], uint64_t da, uint64_t db) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
        "%32, %33, 1, 1, 1, 0, 0;\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
          "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]),
          "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]),
          "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "l"(da), "l"(db)
        : "memory");
}
__device__ __forceinline__ void wgmma_m64n64k16_tb1(float (&d)[32], uint64_t da, uint64_t db) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
        "%32, %33, 1, 1, 1, 0, 1;\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
          "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]),
          "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]),
          "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "l"(da), "l"(db)
        : "memory");
}

/* ============================ TMA / mbarrier primitives ======================================= */
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void mbar_init(uint64_t* b, int cnt) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(b)), "r"(cnt) : "memory");
}
__device__ __forceinline__ void mbar_expect(uint64_t* b, int bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(b)),
                 "r"(bytes)
                 : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* b) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, int phase) {
    asm volatile("{ .reg .pred p; TW%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;"
                 " @!p bra TW%=; }" ::"r"(smem_u32(b)),
                 "r"(phase)
                 : "memory");
}
/* 3-D TMA: box {64 elems of HD, ROWS of seq, 1 head}. Coordinates are ELEMENT indices. */
__device__ __forceinline__ void tma3d(uint32_t dst, const CUtensorMap* map, int c0, int c1, int c2,
                                      uint32_t bar) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes"
                 " [%0], [%1, {%2, %3, %4}], [%5];" ::"r"(dst),
                 "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(bar)
                 : "memory");
}
/* 2-D TMA, only used by the layout-equality selftest. */
__device__ __forceinline__ void tma2d(uint32_t dst, const CUtensorMap* map, int c0, int c1,
                                      uint32_t bar) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
                 " [%0], [%1, {%2, %3}], [%4];" ::"r"(dst),
                 "l"(map), "r"(c0), "r"(c1), "r"(bar)
                 : "memory");
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_u32(smem)), "l"(gmem)
                 : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_async_wait0() { asm volatile("cp.async.wait_group 0;\n" ::); }

__device__ __forceinline__ float ex2(float x) {
    float r;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}
__device__ __forceinline__ bf16* align1k(void* p) {
    unsigned off = smem_u32(p) & 1023u;
    return (bf16*)((char*)p + ((1024u - off) & 1023u));
}

/* ============================ SELFTEST 1: K-major 128B swizzle ================================
 * C[64][64] = A[64][64] . B[64][64]^T, both operands stored 128B-swizzled IN SOFTWARE (so the
 * descriptor is isolated from TMA). Reproduces hopper_warpspec_prefill.cu's validated recipe at
 * m64n64k16 and at the sub-tile geometry this kernel actually uses. */
__global__ void k_sw128_km_selftest(const bf16* __restrict__ A, const bf16* __restrict__ B,
                                    float* __restrict__ C) {
    extern __shared__ char smem_raw[];
    bf16* As = align1k(smem_raw);
    bf16* Bs = As + 64 * 64;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    for (int i = tid; i < 64 * 64; i += 128) {
        As[sw128_off(i / 64, i % 64)] = A[i];
        Bs[sw128_off(i / 64, i % 64)] = B[i];
    }
    __syncthreads();
    async_proxy_fence();
    float d[32];
#pragma unroll
    for (int i = 0; i < 32; i++) d[i] = 0.0f;
    wgmma_fence();
#pragma unroll
    for (int sub = 0; sub < 4; sub++)
        wgmma_m64n64k16(d, desc_kmajor_sw128(As + sub * 16), desc_kmajor_sw128(Bs + sub * 16));
    wgmma_commit();
    wgmma_wait0();
#pragma unroll
    for (int nb = 0; nb < 8; nb++)
#pragma unroll
        for (int e = 0; e < 2; e++) {
            C[ACC_ROW_A(warp, lane) * 64 + ACC_COL(nb, lane, e)] = d[4 * nb + e];
            C[ACC_ROW_B(warp, lane) * 64 + ACC_COL(nb, lane, e)] = d[4 * nb + 2 + e];
        }
}

/* ============================ SELFTEST 2: MN-major 128B swizzle ===============================
 * C[64][64] = A[64][64] . B2[64][64] with B2 stored [K][N] (N contiguous), 128B-swizzled.
 * This is the descriptor V needs. Both stride variants are run; the smem window is 64 KiB with
 * Bs at offset 16 KiB, so the losing variant reads garbage INSIDE the allocation rather than
 * raising a sticky illegal access that would kill every later case. */
template <int VAR>
__global__ void k_sw128_mn_selftest(const bf16* __restrict__ A, const bf16* __restrict__ B2,
                                    float* __restrict__ C) {
    extern __shared__ char smem_raw[];
    bf16* As = align1k(smem_raw);
    bf16* Bs = As + 8192; /* +16 KiB */
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    for (int i = tid; i < 64 * 64; i += 128) {
        As[sw128_off(i / 64, i % 64)] = A[i];  /* [M][K] K-major   */
        Bs[sw128_off(i / 64, i % 64)] = B2[i]; /* [K][N] MN-major  */
    }
    /* zero the 8 KiB hole between As and Bs, and the 8 KiB tail after Bs, so the losing stride
     * variant reads DEFINED memory inside the allocation instead of faulting */
    for (int i = tid; i < 4096; i += 128) {
        As[4096 + i] = __float2bfloat16(0.f);
        Bs[4096 + i] = __float2bfloat16(0.f);
    }
    __syncthreads();
    async_proxy_fence();
    float d[32];
#pragma unroll
    for (int i = 0; i < 32; i++) d[i] = 0.0f;
    wgmma_fence();
#pragma unroll
    for (int ks = 0; ks < 4; ks++) /* k-slice = 16 rows of B2 = 1024 elements */
        wgmma_m64n64k16_tb1(d, desc_kmajor_sw128(As + ks * 16),
                            desc_mnmajor_sw128<VAR>(Bs + ks * 1024));
    wgmma_commit();
    wgmma_wait0();
#pragma unroll
    for (int nb = 0; nb < 8; nb++)
#pragma unroll
        for (int e = 0; e < 2; e++) {
            C[ACC_ROW_A(warp, lane) * 64 + ACC_COL(nb, lane, e)] = d[4 * nb + e];
            C[ACC_ROW_B(warp, lane) * 64 + ACC_COL(nb, lane, e)] = d[4 * nb + 2 + e];
        }
}

/* ============================ SELFTEST 3: TMA layout == sw128_off =============================
 * Does cp.async.bulk.tensor with CU_TENSOR_MAP_SWIZZLE_128B write exactly the layout that
 * sw128_off() describes (and that selftests 1-2 just validated)? Dump the smem tile linearly. */
__global__ void k_tma_layout(const __grid_constant__ CUtensorMap map, bf16* __restrict__ out,
                             int rows) {
    extern __shared__ char smem_raw[];
    uint64_t* bar = (uint64_t*)smem_raw;
    bf16* t = align1k(smem_raw + 128);
    if (threadIdx.x == 0) {
        mbar_init(bar, 1);
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        mbar_expect(bar, rows * 64 * 2);
        tma2d(smem_u32(t), &map, 0, 0, smem_u32(bar));
    }
    mbar_wait(bar, 0);
    for (int i = threadIdx.x; i < rows * 64; i += blockDim.x) out[i] = t[i];
}

/* ============================ PROBE REFERENCE (verbatim) ======================================
 * Single-buffered cp.async, swizzle-0 core-matrix-packed operands, no KV pipelining, no causal
 * tile skip. This is the 2.82x kernel the whole comparison is against. */
template <int HD, int BQ, int BKV>
__global__ __launch_bounds__(128) void k_probe_flash(const bf16* __restrict__ Q,
                                                     const bf16* __restrict__ K,
                                                     const bf16* __restrict__ V,
                                                     float* __restrict__ O, int seq_q, int seq_kv,
                                                     int n_head, int n_kv_head, float lscale) {
    static_assert(BQ == 64, "one warpgroup m64");
    constexpr int NT = HD / 64, KS0 = HD / 16, KS1 = BKV / 16;
    extern __shared__ char smem_raw[];
    bf16* Qs = (bf16*)smem_raw;
    bf16* Ks = Qs + BQ * HD;
    bf16* Vs = Ks + BKV * HD;
    bf16* Ps = Vs + BKV * HD;

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int h = blockIdx.x, qt = blockIdx.y;
    const int gqa = n_head / n_kv_head, hkv = h / gqa;
    const int q_pos0 = seq_kv - seq_q + qt * BQ;
    const bf16* Qh = Q + ((size_t)h * seq_q + (size_t)qt * BQ) * HD;
    const bf16* Kh = K + (size_t)hkv * seq_kv * HD;
    const bf16* Vh = V + (size_t)hkv * seq_kv * HD;
    const int qrows = (seq_q - qt * BQ < BQ) ? (seq_q - qt * BQ) : BQ;

    for (int i = tid * 8; i < BQ * HD; i += 128 * 8) {
        const int r = i / HD, c = i % HD;
        bf16* dst = Qs + cm_off(r, c, HD);
        if (r < qrows) cp_async16(dst, Qh + (size_t)r * HD + c);
        else *(uint4*)dst = make_uint4(0, 0, 0, 0);
    }
    cp_async_commit();
    cp_async_wait0();
    __syncthreads();

    float Oacc[NT][32];
#pragma unroll
    for (int t = 0; t < NT; t++)
#pragma unroll
        for (int i = 0; i < 32; i++) Oacc[t][i] = 0.0f;

    const int rA = ACC_ROW_A(warp, lane), rB = ACC_ROW_B(warp, lane);
    const int qposA = q_pos0 + rA, qposB = q_pos0 + rB;
    float mA = NEG_BIG, lA = 0.0f, mB = NEG_BIG, lB = 0.0f;
    const int ntiles = (seq_kv + BKV - 1) / BKV;
    for (int t = 0; t < ntiles; t++) {
        const int kvbase = t * BKV;
        const int krows = (seq_kv - kvbase < BKV) ? (seq_kv - kvbase) : BKV;
        for (int i = tid * 8; i < BKV * HD; i += 128 * 8) {
            const int r = i / HD, c = i % HD;
            const int off = cm_off(r, c, HD);
            if (r < krows) {
                cp_async16(Ks + off, Kh + (size_t)(kvbase + r) * HD + c);
                cp_async16(Vs + off, Vh + (size_t)(kvbase + r) * HD + c);
            } else {
                *(uint4*)(Ks + off) = make_uint4(0, 0, 0, 0);
                *(uint4*)(Vs + off) = make_uint4(0, 0, 0, 0);
            }
        }
        cp_async_commit();
        cp_async_wait0();
        __syncthreads();
        async_proxy_fence();

        float S[32];
#pragma unroll
        for (int i = 0; i < 32; i++) S[i] = 0.0f;
        wgmma_fence();
#pragma unroll 1
        for (int ks = 0; ks < KS0; ks++)
            wgmma_m64n64k16(S, make_smem_desc0(CM_KSLICE(Qs, ks), HD),
                            make_smem_desc0(CM_KSLICE(Ks, ks), HD));
        wgmma_commit();
        wgmma_wait0();

        float mxA = NEG_BIG, mxB = NEG_BIG;
#pragma unroll
        for (int nb = 0; nb < BKV / 8; nb++)
#pragma unroll
            for (int e = 0; e < 2; e++) {
                const int kvp = kvbase + ACC_COL(nb, lane, e);
                const bool ok = (kvp < seq_kv);
                const float a = (ok && kvp <= qposA) ? S[4 * nb + e] * lscale : NEG_BIG;
                const float b = (ok && kvp <= qposB) ? S[4 * nb + 2 + e] * lscale : NEG_BIG;
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
        const float cA = ex2(mA - mnA), cB = ex2(mB - mnB);
        mA = mnA;
        mB = mnB;
        float sA = 0.0f, sB = 0.0f;
#pragma unroll
        for (int nb = 0; nb < BKV / 8; nb++) {
            const float pa0 = ex2(S[4 * nb + 0] - mnA), pa1 = ex2(S[4 * nb + 1] - mnA);
            const float pb0 = ex2(S[4 * nb + 2] - mnB), pb1 = ex2(S[4 * nb + 3] - mnB);
            sA += pa0 + pa1;
            sB += pb0 + pb1;
            const int col0 = ACC_COL(nb, lane, 0);
            *(__nv_bfloat162*)(Ps + cm_off(rA, col0, BKV)) = __floats2bfloat162_rn(pa0, pa1);
            *(__nv_bfloat162*)(Ps + cm_off(rB, col0, BKV)) = __floats2bfloat162_rn(pb0, pb1);
        }
        sA += __shfl_xor_sync(0xffffffffu, sA, 1);
        sA += __shfl_xor_sync(0xffffffffu, sA, 2);
        sB += __shfl_xor_sync(0xffffffffu, sB, 1);
        sB += __shfl_xor_sync(0xffffffffu, sB, 2);
        lA = lA * cA + sA;
        lB = lB * cB + sB;
#pragma unroll
        for (int nt = 0; nt < NT; nt++)
#pragma unroll
            for (int nb = 0; nb < 8; nb++)
#pragma unroll
                for (int e = 0; e < 2; e++) {
                    Oacc[nt][4 * nb + e] *= cA;
                    Oacc[nt][4 * nb + 2 + e] *= cB;
                }
        __syncthreads();
        async_proxy_fence();
        wgmma_fence();
#pragma unroll
        for (int nt = 0; nt < NT; nt++)
#pragma unroll 1
            for (int ks = 0; ks < KS1; ks++)
                wgmma_m64n64k16_tb1(Oacc[nt], make_smem_desc0(CM_KSLICE(Ps, ks), BKV),
                                    make_desc(Vs + ks * 16 * HD + nt * 512, 16 * HD, 128, 0));
        wgmma_commit();
        wgmma_wait0();
        __syncthreads();
    }
    const float iA = (lA > 0.0f) ? 1.0f / lA : 0.0f;
    const float iB = (lB > 0.0f) ? 1.0f / lB : 0.0f;
    float* Oh = O + ((size_t)h * seq_q + (size_t)qt * BQ) * HD;
#pragma unroll
    for (int nt = 0; nt < NT; nt++)
#pragma unroll
        for (int nb = 0; nb < 8; nb++)
#pragma unroll
            for (int e = 0; e < 2; e++) {
                const int hd = nt * 64 + ACC_COL(nb, lane, e);
                if (rA < qrows) Oh[(size_t)rA * HD + hd] = Oacc[nt][4 * nb + e] * iA;
                if (rB < qrows) Oh[(size_t)rB * HD + hd] = Oacc[nt][4 * nb + 2 + e] * iB;
            }
}

/* ============================ TMA + WARP-SPECIALIZED FLASH PREFILL ============================
 * MODE 0 = uniform (128 thr, thread 0 is the TMA producer inside the same warpgroup)
 * MODE 1 = warp specialized (256 thr: producer WG + consumer WG), no register control
 * MODE 2 = MODE 1 + setmaxnreg.dec(PREG) on the producer / .inc(2*128-PREG) on the consumer
 * CSKIP  = 1 skips KV tiles entirely above the causal diagonal (never staged, never computed)
 * MNV    = MN-major descriptor variant for V (selftest 2 picks it; 0 is the shipped answer)
 *
 * smem:  [barriers][Q: HD/64 sub-tiles][K ring: NS x HD/64][V ring: NS x HD/64][Ps]
 * Every sub-tile is [rows][64] bf16, 128B-swizzled by the TMA hardware, 1024 B aligned. */
template <int HD, int BQ, int BKV, int NS, int MODE, int PREG, int CSKIP, int MNV>
__device__ __forceinline__ void tma_flash_body(const CUtensorMap* qm, const CUtensorMap* km,
                                               const CUtensorMap* vm, float* __restrict__ O,
                                               int seq_q, int seq_kv, int n_head, int n_kv_head,
                                               float lscale) {
    static_assert(BQ == 64, "one warpgroup m64");
    static_assert(HD % 64 == 0, "128B swizzle fixes the sub-tile at 64 bf16");
    static_assert(BKV % 16 == 0, "k16 steps");
    constexpr int NT = HD / 64;          /* O accumulator n64 tiles == V sub-tiles == HD chunks */
    constexpr int QSUB = BQ * 64;        /* elements per Q sub-tile   */
    constexpr int KSUB = BKV * 64;       /* elements per K/V sub-tile */
    constexpr int STAGE = NT * KSUB;     /* elements per K (or V) ring stage */
    constexpr int QBYTES = BQ * HD * 2;
    constexpr int KVBYTES = 2 * BKV * HD * 2;
    constexpr int CREG = 2 * 128 - PREG;

    extern __shared__ char smem_raw[];
    uint64_t* bfull = (uint64_t*)smem_raw;
    uint64_t* bempty = bfull + NS;
    uint64_t* bq = bempty + NS;
    bf16* Qs = align1k(smem_raw + 1024);
    bf16* Ks = Qs + BQ * HD;
    bf16* Vs = Ks + NS * STAGE;
    bf16* Ps = Vs + NS * STAGE;

    const int tid = threadIdx.x;
    if (tid < 2 * NS + 1) mbar_init(bfull + tid, 1);
    async_proxy_fence(); /* mbarrier state must be visible to the async (TMA) proxy */
    __syncthreads();

    const int h = blockIdx.x, qt = blockIdx.y;
    const int gqa = n_head / n_kv_head, hkv = h / gqa;
    const int q_pos0 = seq_kv - seq_q + qt * BQ;
    const int qrows = (seq_q - qt * BQ < BQ) ? (seq_q - qt * BQ) : BQ;
    const int ntiles_all = (seq_kv + BKV - 1) / BKV;
    int ntiles = ntiles_all;
    if constexpr (CSKIP) {
        /* last query row of this tile sits at q_pos0+qrows-1; every KV tile strictly beyond it
         * is fully masked, so it is never staged and never computed. */
        const int need = (q_pos0 + qrows - 1) / BKV + 1;
        ntiles = need < ntiles_all ? need : ntiles_all;
        if (ntiles < 1) ntiles = 1;
    }

    /* ---------------- producer ---------------- */
    if constexpr (MODE != 0) {
        if (tid < 128) {
            if constexpr (MODE == 2)
                asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(PREG));
            if (tid == 0) {
                mbar_expect(bq, QBYTES);
#pragma unroll
                for (int kt = 0; kt < NT; kt++)
                    tma3d(smem_u32(Qs + kt * QSUB), qm, kt * 64, qt * BQ, h, smem_u32(bq));
                for (int t = 0; t < ntiles; t++) {
                    const int s = t % NS;
                    if (t >= NS) mbar_wait(bempty + s, ((t / NS) + 1) & 1);
                    mbar_expect(bfull + s, KVBYTES);
#pragma unroll
                    for (int kt = 0; kt < NT; kt++) {
                        tma3d(smem_u32(Ks + s * STAGE + kt * KSUB), km, kt * 64, t * BKV, hkv,
                              smem_u32(bfull + s));
                        tma3d(smem_u32(Vs + s * STAGE + kt * KSUB), vm, kt * 64, t * BKV, hkv,
                              smem_u32(bfull + s));
                    }
                }
            }
            return;
        }
        if constexpr (MODE == 2) asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(CREG));
    } else {
        if (tid == 0) {
            mbar_expect(bq, QBYTES);
#pragma unroll
            for (int kt = 0; kt < NT; kt++)
                tma3d(smem_u32(Qs + kt * QSUB), qm, kt * 64, qt * BQ, h, smem_u32(bq));
            for (int s = 0; s < NS && s < ntiles; s++) {
                mbar_expect(bfull + s, KVBYTES);
#pragma unroll
                for (int kt = 0; kt < NT; kt++) {
                    tma3d(smem_u32(Ks + s * STAGE + kt * KSUB), km, kt * 64, s * BKV, hkv,
                          smem_u32(bfull + s));
                    tma3d(smem_u32(Vs + s * STAGE + kt * KSUB), vm, kt * 64, s * BKV, hkv,
                          smem_u32(bfull + s));
                }
            }
        }
    }

    /* ---------------- consumer warpgroup ---------------- */
    const int ctid = (MODE == 0) ? tid : tid - 128;
    const int warp = ctid >> 5, lane = ctid & 31;

    float Oacc[NT][32];
#pragma unroll
    for (int t = 0; t < NT; t++)
#pragma unroll
        for (int i = 0; i < 32; i++) Oacc[t][i] = 0.0f;

    const int rA = ACC_ROW_A(warp, lane), rB = ACC_ROW_B(warp, lane);
    const int qposA = q_pos0 + rA, qposB = q_pos0 + rB;
    float mA = NEG_BIG, lA = 0.0f, mB = NEG_BIG, lB = 0.0f;

    mbar_wait(bq, 0);

    for (int t = 0; t < ntiles; t++) {
        const int s = t % NS;
        const int kvbase = t * BKV;
        mbar_wait(bfull + s, (t / NS) & 1);
        const bf16* Kc = Ks + s * STAGE;
        const bf16* Vc = Vs + s * STAGE;

        /* --- GEMM0: S[BQ][BKV] = Q . K^T, both operands K-major 128B-swizzled --- */
        float S[32];
#pragma unroll
        for (int i = 0; i < 32; i++) S[i] = 0.0f;
        wgmma_fence();
#pragma unroll 1
        for (int kt = 0; kt < NT; kt++)
#pragma unroll
            for (int sub = 0; sub < 4; sub++)
                wgmma_m64n64k16(S, desc_kmajor_sw128(Qs + kt * QSUB + sub * 16),
                                desc_kmajor_sw128(Kc + kt * KSUB + sub * 16));
        wgmma_commit();
        wgmma_wait0();

        /* --- scale + causal mask IN THE FRAGMENT (finding (a): no smem round trip) --- */
        float mxA = NEG_BIG, mxB = NEG_BIG;
#pragma unroll
        for (int nb = 0; nb < BKV / 8; nb++)
#pragma unroll
            for (int e = 0; e < 2; e++) {
                const int kvp = kvbase + ACC_COL(nb, lane, e);
                const bool ok = (kvp < seq_kv);
                const float a = (ok && kvp <= qposA) ? S[4 * nb + e] * lscale : NEG_BIG;
                const float b = (ok && kvp <= qposB) ? S[4 * nb + 2 + e] * lscale : NEG_BIG;
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
        const float cA = ex2(mA - mnA), cB = ex2(mB - mnB);
        mA = mnA;
        mB = mnB;

        float sA = 0.0f, sB = 0.0f;
#pragma unroll
        for (int nb = 0; nb < BKV / 8; nb++) {
            const float pa0 = ex2(S[4 * nb + 0] - mnA), pa1 = ex2(S[4 * nb + 1] - mnA);
            const float pb0 = ex2(S[4 * nb + 2] - mnB), pb1 = ex2(S[4 * nb + 3] - mnB);
            sA += pa0 + pa1;
            sB += pb0 + pb1;
            const int col0 = ACC_COL(nb, lane, 0);
            *(__nv_bfloat162*)(Ps + cm_off(rA, col0, BKV)) = __floats2bfloat162_rn(pa0, pa1);
            *(__nv_bfloat162*)(Ps + cm_off(rB, col0, BKV)) = __floats2bfloat162_rn(pb0, pb1);
        }
        sA += __shfl_xor_sync(0xffffffffu, sA, 1);
        sA += __shfl_xor_sync(0xffffffffu, sA, 2);
        sB += __shfl_xor_sync(0xffffffffu, sB, 1);
        sB += __shfl_xor_sync(0xffffffffu, sB, 2);
        lA = lA * cA + sA;
        lB = lB * cB + sB;

#pragma unroll
        for (int nt = 0; nt < NT; nt++)
#pragma unroll
            for (int nb = 0; nb < 8; nb++)
#pragma unroll
                for (int e = 0; e < 2; e++) {
                    Oacc[nt][4 * nb + e] *= cA;
                    Oacc[nt][4 * nb + 2 + e] *= cB;
                }

        if constexpr (MODE == 0) __syncthreads();
        else asm volatile("bar.sync 1, 128;" ::: "memory");
        async_proxy_fence();

        /* --- GEMM1: O += P . V.  A = Ps (swizzle-0 packed, K-major in BKV),
         * B = V sub-tile nt (MN-MAJOR 128B-swizzled, trans-b=1) — finding (b), no transpose.
         * k-slice ks = 16 rows of the [BKV][64] sub-tile = +1024 elements = +2048 B. --- */
        wgmma_fence();
#pragma unroll
        for (int nt = 0; nt < NT; nt++)
#pragma unroll 1
            for (int ks = 0; ks < BKV / 16; ks++)
                wgmma_m64n64k16_tb1(Oacc[nt], make_smem_desc0(CM_KSLICE(Ps, ks), BKV),
                                    desc_mnmajor_sw128<MNV>(Vc + nt * KSUB + ks * 1024));
        wgmma_commit();
        wgmma_wait0();

        if constexpr (MODE == 0) {
            __syncthreads();
            if (tid == 0 && t + NS < ntiles) {
                mbar_expect(bfull + s, KVBYTES);
                const int tn = t + NS;
#pragma unroll
                for (int kt = 0; kt < NT; kt++) {
                    tma3d(smem_u32(Ks + s * STAGE + kt * KSUB), km, kt * 64, tn * BKV, hkv,
                          smem_u32(bfull + s));
                    tma3d(smem_u32(Vs + s * STAGE + kt * KSUB), vm, kt * 64, tn * BKV, hkv,
                          smem_u32(bfull + s));
                }
            }
        } else {
            asm volatile("bar.sync 1, 128;" ::: "memory");
            if (ctid == 0) mbar_arrive(bempty + s);
        }
    }

    const float iA = (lA > 0.0f) ? 1.0f / lA : 0.0f;
    const float iB = (lB > 0.0f) ? 1.0f / lB : 0.0f;
    float* Oh = O + ((size_t)h * seq_q + (size_t)qt * BQ) * HD;
#pragma unroll
    for (int nt = 0; nt < NT; nt++)
#pragma unroll
        for (int nb = 0; nb < 8; nb++)
#pragma unroll
            for (int e = 0; e < 2; e++) {
                const int hd = nt * 64 + ACC_COL(nb, lane, e);
                if (rA < qrows) Oh[(size_t)rA * HD + hd] = Oacc[nt][4 * nb + e] * iA;
                if (rB < qrows) Oh[(size_t)rB * HD + hd] = Oacc[nt][4 * nb + 2 + e] * iB;
            }
}

#define TMA_ARGS                                                                                   \
    const __grid_constant__ CUtensorMap qm, const __grid_constant__ CUtensorMap km,                \
        const __grid_constant__ CUtensorMap vm, float *O, int sq, int sk, int nh, int nkv,         \
        float ls
#define TMA_CALL(MODE, PREG) tma_flash_body<HD, BQ, BKV, NS, MODE, PREG, CSKIP, MNV>(&qm, &km, &vm, O, sq, sk, nh, nkv, ls)

template <int HD, int BQ, int BKV, int NS, int CSKIP, int MNV>
__global__ __launch_bounds__(128) void k_tma_uni(TMA_ARGS) {
    TMA_CALL(0, 0);
}
template <int HD, int BQ, int BKV, int NS, int CSKIP, int MNV>
__global__ void k_tma_ws(TMA_ARGS) {
    TMA_CALL(1, 0);
}
/* __maxnreg__, NOT __launch_bounds__: ptxas silently drops the setmaxnreg effect with the latter. */
template <int HD, int BQ, int BKV, int NS, int PREG, int CSKIP, int MNV>
__global__ void __maxnreg__(128) k_tma_wsr(TMA_ARGS) {
    TMA_CALL(2, PREG);
}

/* ============================ host: oracle, harness =========================================== */
static float bf2f(bf16 x) { return __bfloat162float(x); }
static bf16 f2bf(float x) { return __float2bfloat16(x); }

static void cpu_oracle(const std::vector<bf16>& Q, const std::vector<bf16>& K,
                       const std::vector<bf16>& V, std::vector<float>& O, int HD, int seq_q,
                       int seq_kv, int nh, int nkv) {
    const float scale = 1.0f / sqrtf((float)HD);
    const int gqa = nh / nkv;
    O.assign((size_t)nh * seq_q * HD, 0.0f);
    std::vector<float> s(seq_kv);
    for (int h = 0; h < nh; h++) {
        const int hkv = h / gqa;
        for (int i = 0; i < seq_q; i++) {
            const int qpos = seq_kv - seq_q + i;
            const bf16* qr = &Q[((size_t)h * seq_q + i) * HD];
            float mx = -INFINITY;
            const int jmax = qpos < seq_kv - 1 ? qpos : seq_kv - 1;
            for (int j = 0; j <= jmax; j++) {
                const bf16* kr = &K[((size_t)hkv * seq_kv + j) * HD];
                float d = 0.0f;
                for (int c = 0; c < HD; c++) d += bf2f(qr[c]) * bf2f(kr[c]);
                s[j] = d * scale;
                if (s[j] > mx) mx = s[j];
            }
            float sum = 0.0f;
            for (int j = 0; j <= jmax; j++) {
                s[j] = expf(s[j] - mx);
                sum += s[j];
            }
            const float inv = sum > 0.f ? 1.0f / sum : 0.0f;
            float* orow = &O[((size_t)h * seq_q + i) * HD];
            for (int j = 0; j <= jmax; j++) {
                const float p = s[j] * inv;
                if (p == 0.0f) continue;
                const bf16* vr = &V[((size_t)hkv * seq_kv + j) * HD];
                for (int c = 0; c < HD; c++) orow[c] += p * bf2f(vr[c]);
            }
        }
    }
}

static double rel_l2(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); i++) {
        const double d = (double)a[i] - (double)b[i];
        num += d * d;
        den += (double)b[i] * (double)b[i];
    }
    return den > 0 ? sqrt(num / den) : sqrt(num);
}
static uint32_t rs = 12345u;
static float frand() {
    rs = rs * 1664525u + 1013904223u;
    return ((float)((rs >> 8) & 0xFFFF) / 32768.0f - 1.0f);
}

/* 3-D tensor map over a [heads][seq][HD] bf16 tensor, box {64, ROWS, 1}, 128 B swizzle.
 * The head dimension is a REAL tensor dim (not folded into rows) so that the per-head sequence
 * end is an out-of-bounds boundary and TMA zero-fills the ragged tail. */
static void make_map3d(CUtensorMap* map, const void* base, int HD, int seq, int heads, int rows) {
    memset(map, 0, sizeof(*map));
    uint64_t gdim[3] = {(uint64_t)HD, (uint64_t)seq, (uint64_t)heads};
    uint64_t gstr[2] = {(uint64_t)HD * 2, (uint64_t)HD * (uint64_t)seq * 2};
    uint32_t bdim[3] = {64u, (uint32_t)rows, 1u};
    uint32_t estr[3] = {1, 1, 1};
    CKD(cuTensorMapEncodeTiled(map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, (void*)base, gdim, gstr,
                               bdim, estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
                               CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

/* MEASURED by selftest 2 on H100 NVL: for an MN-major operand under 128B SWIZZLE the descriptor
 * strides are IDENTICAL to the K-major ones (LBO=16, SBO=1024). This is the opposite of the
 * NO-SWIZZLE case, where wgmma_flash_prefill_probe.cu had to swap them (LBO=16*W, SBO=128).
 * Under a swizzle mode the layout is self-describing to the hardware, so only trans-b changes. */
static constexpr int MNV_SHIP = 1;
static int g_mnv = 0; /* MN-major descriptor variant chosen by selftest 2 */

/* ---- selftests ---- */
static int run_selftests() {
    int ok = 1;
    std::vector<bf16> A(64 * 64), B(64 * 64);
    for (auto& x : A) x = f2bf(frand());
    for (auto& x : B) x = f2bf(frand());
    std::vector<float> refT(64 * 64, 0.f), refN(64 * 64, 0.f);
    for (int i = 0; i < 64; i++)
        for (int j = 0; j < 64; j++) {
            float dt = 0, dn = 0;
            for (int k = 0; k < 64; k++) {
                dt += bf2f(A[i * 64 + k]) * bf2f(B[j * 64 + k]);
                dn += bf2f(A[i * 64 + k]) * bf2f(B[k * 64 + j]);
            }
            refT[i * 64 + j] = dt;
            refN[i * 64 + j] = dn;
        }
    bf16 *dA, *dB;
    float* dC;
    CK(cudaMalloc(&dA, A.size() * 2));
    CK(cudaMalloc(&dB, B.size() * 2));
    CK(cudaMalloc(&dC, 64 * 64 * 4));
    CK(cudaMemcpy(dA, A.data(), A.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, B.data(), B.size() * 2, cudaMemcpyHostToDevice));
    std::vector<float> got(64 * 64);

    CK(cudaFuncSetAttribute(k_sw128_km_selftest, cudaFuncAttributeMaxDynamicSharedMemorySize,
                            32768));
    CK(cudaMemset(dC, 0, 64 * 64 * 4));
    k_sw128_km_selftest<<<1, 128, 32768>>>(dA, dB, dC);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(got.data(), dC, 64 * 64 * 4, cudaMemcpyDeviceToHost));
    double e = rel_l2(got, refT);
    printf("[selftest 1] K-major 128B-swizzle  A.B^T  m64n64k16  relL2 = %.3e  %s\n", e,
           e < 3e-3 ? "PASS" : "FAIL");
    if (e >= 3e-3) ok = 0;

    double best = 1e30;
    for (int v = 0; v < 2; v++) {
        CK(cudaMemset(dC, 0, 64 * 64 * 4));
        if (v == 0) {
            CK(cudaFuncSetAttribute(k_sw128_mn_selftest<0>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, 65536));
            k_sw128_mn_selftest<0><<<1, 128, 65536>>>(dA, dB, dC);
        } else {
            CK(cudaFuncSetAttribute(k_sw128_mn_selftest<1>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, 65536));
            k_sw128_mn_selftest<1><<<1, 128, 65536>>>(dA, dB, dC);
        }
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(got.data(), dC, 64 * 64 * 4, cudaMemcpyDeviceToHost));
        e = rel_l2(got, refN);
        printf("[selftest 2] MN-major 128B-swizzle trans-b=1 variant %d (%s) relL2 = %.3e  %s\n", v,
               v == 0 ? "LBO=1024,SBO=16" : "LBO=16,SBO=1024", e, e < 3e-3 ? "PASS" : "fail");
        if (e < best) {
            best = e;
            g_mnv = v;
        }
    }
    if (best >= 3e-3) ok = 0;
    printf("[selftest 2] -> MN-major variant %d selected\n", g_mnv);

    /* selftest 3: TMA SWIZZLE_128B writes exactly sw128_off() */
    {
        const int ROWS = 64;
        std::vector<bf16> T(ROWS * 64);
        for (int i = 0; i < ROWS * 64; i++) T[i] = f2bf((float)(i % 251) - 125.0f);
        bf16* dT;
        CK(cudaMalloc(&dT, T.size() * 2));
        CK(cudaMemcpy(dT, T.data(), T.size() * 2, cudaMemcpyHostToDevice));
        CUtensorMap map;
        memset(&map, 0, sizeof(map));
        uint64_t gdim[2] = {64ull, (uint64_t)ROWS};
        uint64_t gstr[1] = {128ull};
        uint32_t bdim[2] = {64u, (uint32_t)ROWS};
        uint32_t estr[2] = {1, 1};
        CKD(cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, dT, gdim, gstr, bdim,
                                   estr, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                   CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                   CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
        bf16* dOut;
        CK(cudaMalloc(&dOut, T.size() * 2));
        CK(cudaMemset(dOut, 0, T.size() * 2));
        const int sm = 1024 + ROWS * 64 * 2;
        CK(cudaFuncSetAttribute(k_tma_layout, cudaFuncAttributeMaxDynamicSharedMemorySize, sm));
        k_tma_layout<<<1, 128, sm>>>(map, dOut, ROWS);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        std::vector<bf16> H(T.size());
        CK(cudaMemcpy(H.data(), dOut, T.size() * 2, cudaMemcpyDeviceToHost));
        int bad = 0;
        for (int r = 0; r < ROWS; r++)
            for (int c = 0; c < 64; c++)
                if (bf2f(H[sw128_off(r, c)]) != bf2f(T[r * 64 + c])) bad++;
        printf("[selftest 3] TMA SWIZZLE_128B layout == sw128_off(): %d/%d mismatches  %s\n", bad,
               ROWS * 64, bad == 0 ? "PASS" : "FAIL");
        if (bad) ok = 0;
        CK(cudaFree(dT));
        CK(cudaFree(dOut));
    }
    CK(cudaFree(dA));
    CK(cudaFree(dB));
    CK(cudaFree(dC));
    return ok;
}

/* ---- smem sizes ---- */
template <int HD, int BQ, int BKV> static constexpr size_t probe_smem() {
    return (size_t)(BQ * HD + 2 * BKV * HD + BQ * BKV) * 2;
}
template <int HD, int BQ, int BKV, int NS> static constexpr size_t tma_smem() {
    return 2048 + (size_t)(BQ * HD + 2 * NS * BKV * HD + BQ * BKV) * 2;
}

/* ---- one measured variant ---- */
struct Res {
    char name[48];
    double ms;
    int regs, spill, threads, occ;
    size_t smem;
    double relL2;
};

template <int HD, int BQ, int BKV, int NS, int VMODE, int PREG, int CSKIP>
static Res run_variant(const char* label, const CUtensorMap& qm, const CUtensorMap& km,
                       const CUtensorMap& vm, float* dO, int seq_q, int seq_kv, int nh, int nkv,
                       const std::vector<float>* ref, int iters, int warm) {
    /* VMODE 0 = uni, 1 = ws, 2 = wsr; MNV is fixed to 0 (selftest-selected, asserted at startup) */
    const size_t sm = tma_smem<HD, BQ, BKV, NS>();
    const int threads = (VMODE == 0) ? 128 : 256;
    void* fn;
    if constexpr (VMODE == 0) fn = (void*)k_tma_uni<HD, BQ, BKV, NS, CSKIP, MNV_SHIP>;
    else if constexpr (VMODE == 1) fn = (void*)k_tma_ws<HD, BQ, BKV, NS, CSKIP, MNV_SHIP>;
    else fn = (void*)k_tma_wsr<HD, BQ, BKV, NS, PREG, CSKIP, MNV_SHIP>;
    CK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm));
    cudaFuncAttributes fa;
    CK(cudaFuncGetAttributes(&fa, fn));
    int occ = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, threads, sm));

    const float lscale = (1.0f / sqrtf((float)HD)) * 1.4426950408889634f;
    dim3 grid(nh, (seq_q + BQ - 1) / BQ);
    auto launch = [&]() {
        if constexpr (VMODE == 0)
            k_tma_uni<HD, BQ, BKV, NS, CSKIP, MNV_SHIP>
                <<<grid, threads, sm>>>(qm, km, vm, dO, seq_q, seq_kv, nh, nkv, lscale);
        else if constexpr (VMODE == 1)
            k_tma_ws<HD, BQ, BKV, NS, CSKIP, MNV_SHIP>
                <<<grid, threads, sm>>>(qm, km, vm, dO, seq_q, seq_kv, nh, nkv, lscale);
        else
            k_tma_wsr<HD, BQ, BKV, NS, PREG, CSKIP, MNV_SHIP>
                <<<grid, threads, sm>>>(qm, km, vm, dO, seq_q, seq_kv, nh, nkv, lscale);
    };

    Res r;
    snprintf(r.name, sizeof(r.name), "%s", label);
    r.regs = fa.numRegs;
    r.spill = (int)fa.localSizeBytes;
    r.threads = threads;
    r.smem = sm;
    r.occ = occ;
    r.relL2 = -1;
    r.ms = 0;

    if (ref) {
        CK(cudaMemset(dO, 0, (size_t)nh * seq_q * HD * 4));
        launch();
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        std::vector<float> got((size_t)nh * seq_q * HD);
        CK(cudaMemcpy(got.data(), dO, got.size() * 4, cudaMemcpyDeviceToHost));
        r.relL2 = rel_l2(got, *ref);
    }
    if (iters > 0) {
        cudaEvent_t a, b;
        CK(cudaEventCreate(&a));
        CK(cudaEventCreate(&b));
        for (int i = 0; i < warm; i++) launch();
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(a));
        for (int i = 0; i < iters; i++) launch();
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        float ms;
        CK(cudaEventElapsedTime(&ms, a, b));
        r.ms = ms / iters;
        CK(cudaEventDestroy(a));
        CK(cudaEventDestroy(b));
    }
    return r;
}

/* ---- correctness sweep over all TMA variants ---- */
struct Case {
    int seq_q, seq_kv, nh, nkv;
};

template <int HD, int BQ, int BKV> static int run_case(const Case& c, int& ncase, double& worst) {
    const int seq_q = c.seq_q, seq_kv = c.seq_kv, nh = c.nh, nkv = c.nkv;
    std::vector<bf16> Q((size_t)nh * seq_q * HD), K((size_t)nkv * seq_kv * HD),
        V((size_t)nkv * seq_kv * HD);
    for (auto& x : Q) x = f2bf(frand());
    for (auto& x : K) x = f2bf(frand());
    for (auto& x : V) x = f2bf(frand());
    std::vector<float> ref;
    cpu_oracle(Q, K, V, ref, HD, seq_q, seq_kv, nh, nkv);

    bf16 *dQ, *dK, *dV;
    float* dO;
    CK(cudaMalloc(&dQ, Q.size() * 2));
    CK(cudaMalloc(&dK, K.size() * 2));
    CK(cudaMalloc(&dV, V.size() * 2));
    CK(cudaMalloc(&dO, (size_t)nh * seq_q * HD * 4));
    CK(cudaMemcpy(dQ, Q.data(), Q.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dK, K.data(), K.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dV, V.data(), V.size() * 2, cudaMemcpyHostToDevice));

    CUtensorMap qm, km, vm;
    make_map3d(&qm, dQ, HD, seq_q, nh, BQ);
    make_map3d(&km, dK, HD, seq_kv, nkv, BKV);
    make_map3d(&vm, dV, HD, seq_kv, nkv, BKV);

    constexpr int NS = 2;
    Res a = run_variant<HD, BQ, BKV, NS, 0, 0, 0>("uni", qm, km, vm, dO, seq_q, seq_kv, nh, nkv,
                                                  &ref, 0, 0);
    Res b = run_variant<HD, BQ, BKV, NS, 0, 0, 1>("uni+cskip", qm, km, vm, dO, seq_q, seq_kv, nh,
                                                  nkv, &ref, 0, 0);
    Res cc = run_variant<HD, BQ, BKV, NS, 1, 0, 1>("ws+cskip", qm, km, vm, dO, seq_q, seq_kv, nh,
                                                   nkv, &ref, 0, 0);
    Res d = run_variant<HD, BQ, BKV, NS, 2, 32, 1>("wsr+cskip", qm, km, vm, dO, seq_q, seq_kv, nh,
                                                   nkv, &ref, 0, 0);
    const double e = fmax(fmax(a.relL2, b.relL2), fmax(cc.relL2, d.relL2));
    const int ok = e < 3e-3;
    printf("[case] HD=%3d Bq=%3d Tkv=%-5d nh=%d nkv=%d (gqa=%d) | uni %.2e  uni+cs %.2e  ws+cs "
           "%.2e  wsr+cs %.2e | %s\n",
           HD, seq_q, seq_kv, nh, nkv, nh / nkv, a.relL2, b.relL2, cc.relL2, d.relL2,
           ok ? "PASS" : "FAIL");
    ncase++;
    if (e > worst) worst = e;
    CK(cudaFree(dQ));
    CK(cudaFree(dK));
    CK(cudaFree(dV));
    CK(cudaFree(dO));
    return ok;
}

/* ---- benchmark ---- */
static double causal_pairs(int seq_q, int seq_kv, int BQ, int BKV, int nh, int skip) {
    /* number of (q,kv) element pairs actually multiplied, at TILE granularity when skip=1 */
    const int qt_n = (seq_q + BQ - 1) / BQ, kt_n = (seq_kv + BKV - 1) / BKV;
    double pairs = 0;
    for (int qt = 0; qt < qt_n; qt++) {
        const int qrows = (seq_q - qt * BQ < BQ) ? (seq_q - qt * BQ) : BQ;
        const int q_pos0 = seq_kv - seq_q + qt * BQ;
        int nt = kt_n;
        if (skip) {
            int need = (q_pos0 + qrows - 1) / BKV + 1;
            nt = need < kt_n ? need : kt_n;
            if (nt < 1) nt = 1;
        }
        pairs += (double)BQ * BKV * nt;
    }
    return pairs * nh;
}

template <int HD, int BQ, int BKV> static void bench(int seq_q, int seq_kv, int nh, int nkv,
                                                     const char* tag, int iters, int warm) {
    std::vector<bf16> Q((size_t)nh * seq_q * HD), K((size_t)nkv * seq_kv * HD),
        V((size_t)nkv * seq_kv * HD);
    for (auto& x : Q) x = f2bf(frand());
    for (auto& x : K) x = f2bf(frand());
    for (auto& x : V) x = f2bf(frand());
    bf16 *dQ, *dK, *dV;
    float* dO;
    CK(cudaMalloc(&dQ, Q.size() * 2));
    CK(cudaMalloc(&dK, K.size() * 2));
    CK(cudaMalloc(&dV, V.size() * 2));
    CK(cudaMalloc(&dO, (size_t)nh * seq_q * HD * 4));
    CK(cudaMemcpy(dQ, Q.data(), Q.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dK, K.data(), K.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dV, V.data(), V.size() * 2, cudaMemcpyHostToDevice));
    CUtensorMap qm, km, vm;
    make_map3d(&qm, dQ, HD, seq_q, nh, BQ);
    make_map3d(&km, dK, HD, seq_kv, nkv, BKV);
    make_map3d(&vm, dV, HD, seq_kv, nkv, BKV);

    const int nblk = nh * ((seq_q + BQ - 1) / BQ);
    const double dense = 4.0 * (double)seq_q * seq_kv * HD * nh;
    const double useful_skip = 4.0 * HD * causal_pairs(seq_q, seq_kv, BQ, BKV, nh, 1);
    printf("\n=== BENCH %s: HD=%d Bq=%d Tkv=%d nh=%d nkv=%d -> %d blocks (%s), %d iters/%d warm\n",
           tag, HD, seq_q, seq_kv, nh, nkv, nblk, nblk >= 132 ? "FILLS 132 SMs" : "UNDER-FILLED",
           iters, warm);
    printf("%-22s %9s %9s %10s %7s %8s %7s %7s\n", "variant", "ms", "TF/s(den)", "TF/s(real)",
           "smemKiB", "regs", "spillB", "blk/SM");

    /* probe reference */
    {
        const size_t sm = probe_smem<HD, BQ, BKV>();
        CK(cudaFuncSetAttribute((void*)k_probe_flash<HD, BQ, BKV>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm));
        cudaFuncAttributes fa;
        CK(cudaFuncGetAttributes(&fa, (void*)k_probe_flash<HD, BQ, BKV>));
        int occ = 0;
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, (void*)k_probe_flash<HD, BQ, BKV>,
                                                         128, sm));
        const float lscale = (1.0f / sqrtf((float)HD)) * 1.4426950408889634f;
        dim3 grid(nh, (seq_q + BQ - 1) / BQ);
        for (int i = 0; i < warm; i++)
            k_probe_flash<HD, BQ, BKV>
                <<<grid, 128, sm>>>(dQ, dK, dV, dO, seq_q, seq_kv, nh, nkv, lscale);
        CK(cudaDeviceSynchronize());
        cudaEvent_t a, b;
        CK(cudaEventCreate(&a));
        CK(cudaEventCreate(&b));
        CK(cudaEventRecord(a));
        for (int i = 0; i < iters; i++)
            k_probe_flash<HD, BQ, BKV>
                <<<grid, 128, sm>>>(dQ, dK, dV, dO, seq_q, seq_kv, nh, nkv, lscale);
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        float ms;
        CK(cudaEventElapsedTime(&ms, a, b));
        ms /= iters;
        printf("%-22s %9.4f %9.2f %10.2f %7.1f %8d %7d %7d\n", "probe (cp.async,1buf)", ms,
               dense / (ms * 1e-3) / 1e12, dense / (ms * 1e-3) / 1e12, sm / 1024.0, fa.numRegs,
               (int)fa.localSizeBytes, occ);
        CK(cudaEventDestroy(a));
        CK(cudaEventDestroy(b));
    }

    auto show = [&](const Res& r, int skip) {
        const double real = skip ? useful_skip : dense;
        printf("%-22s %9.4f %9.2f %10.2f %7.1f %8d %7d %7d\n", r.name, r.ms,
               dense / (r.ms * 1e-3) / 1e12, real / (r.ms * 1e-3) / 1e12, r.smem / 1024.0, r.regs,
               r.spill, r.occ);
    };

#define RUNV(LBL, NS, VM, PR, CS)                                                                  \
    show(run_variant<HD, BQ, BKV, NS, VM, PR, CS>(LBL, qm, km, vm, dO, seq_q, seq_kv, nh, nkv,     \
                                                  nullptr, iters, warm),                           \
         CS)

    RUNV("tma uni NS=2", 2, 0, 0, 0);
    RUNV("tma uni NS=3", 3, 0, 0, 0);
    RUNV("tma uni NS=4", 4, 0, 0, 0);
    RUNV("tma ws  NS=2", 2, 1, 0, 0);
    RUNV("tma ws  NS=3", 3, 1, 0, 0);
    RUNV("tma ws  NS=4", 4, 1, 0, 0);
    RUNV("tma wsr NS=2 d24", 2, 2, 24, 0);
    RUNV("tma wsr NS=2 d32", 2, 2, 32, 0);
    RUNV("tma wsr NS=2 d40", 2, 2, 40, 0);
    RUNV("tma wsr NS=3 d24", 3, 2, 24, 0);
    RUNV("tma wsr NS=3 d32", 3, 2, 32, 0);
    RUNV("tma wsr NS=4 d32", 4, 2, 32, 0);
    printf("  -- causal tile skip (fewer tiles computed; TF/s(real) is flop-adjusted) --\n");
    RUNV("tma uni NS=2 +cskip", 2, 0, 0, 1);
    RUNV("tma uni NS=3 +cskip", 3, 0, 0, 1);
    RUNV("tma ws  NS=2 +cskip", 2, 1, 0, 1);
    RUNV("tma ws  NS=3 +cskip", 3, 1, 0, 1);
    RUNV("tma wsr NS=2 d24 +cs", 2, 2, 24, 1);
    RUNV("tma wsr NS=2 d32 +cs", 2, 2, 32, 1);
    RUNV("tma wsr NS=3 d24 +cs", 3, 2, 24, 1);
#undef RUNV
    printf("  dense flops %.3f GF, causal-skip flops %.3f GF (%.1f%% of dense)\n", dense / 1e9,
           useful_skip / 1e9, 100.0 * useful_skip / dense);

    CK(cudaFree(dQ));
    CK(cudaFree(dK));
    CK(cudaFree(dV));
    CK(cudaFree(dO));
}

int main(int argc, char** argv) {
    CKD(cuInit(0));
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    printf("GPU: %s  sm_%d%d  SMs=%d  smem/block(optin) %zu KiB  smem/SM %zu KiB\n", p.name,
           p.major, p.minor, p.multiProcessorCount, (size_t)p.sharedMemPerBlockOptin / 1024,
           (size_t)p.sharedMemPerMultiprocessor / 1024);
    if (p.major != 9) {
        printf("RESULT: SKIP (needs sm_90)\n");
        return 0;
    }
    if (!run_selftests()) {
        printf("RESULT: FAIL (layout selftests)\n");
        return 1;
    }
    if (g_mnv != MNV_SHIP) {
        printf("RESULT: FAIL (MN-major variant %d selected but kernels are instantiated with "
               "MNV=%d; rebuild with the other variant)\n",
               g_mnv, MNV_SHIP);
        return 1;
    }

    int ok = 1, ncase = 0;
    double worst = 0;
    ok &= run_case<128, 64, 64>({64, 64, 4, 1}, ncase, worst);
    ok &= run_case<128, 64, 64>({64, 256, 4, 1}, ncase, worst);
    ok &= run_case<128, 64, 64>({64, 1024, 4, 1}, ncase, worst);
    ok &= run_case<128, 64, 64>({64, 64, 4, 2}, ncase, worst);
    ok &= run_case<128, 64, 64>({64, 256, 4, 2}, ncase, worst);
    ok &= run_case<128, 64, 64>({64, 1024, 4, 2}, ncase, worst);
    ok &= run_case<128, 64, 64>({64, 200, 4, 1}, ncase, worst);
    ok &= run_case<128, 64, 64>({64, 200, 4, 2}, ncase, worst);
    ok &= run_case<128, 64, 64>({128, 1024, 4, 2}, ncase, worst);
    ok &= run_case<256, 64, 64>({64, 256, 4, 1}, ncase, worst);
    ok &= run_case<256, 64, 64>({64, 1024, 4, 2}, ncase, worst);
    ok &= run_case<256, 64, 64>({64, 200, 4, 1}, ncase, worst);
    printf("RESULT: %s  (%d cases x 4 variants, worst relL2 = %.3e, gate 3e-3)\n",
           ok ? "PASS" : "FAIL", ncase, worst);
    if (!ok) return 1;

    if (argc > 1 && strcmp(argv[1], "--noperf") == 0) return 0;

    bench<128, 64, 64>(512, 4096, 8, 1, "A (probe's reference shape)", 100, 10);
    bench<128, 64, 64>(512, 4096, 32, 8, "B (machine-filling, GQA 4)", 100, 10);
    bench<128, 64, 64>(512, 16384, 8, 1, "C (long context)", 100, 10);
    bench<128, 64, 64>(4096, 4096, 8, 1, "D (full prefill, causal skip pays)", 50, 5);
    return 0;
}
