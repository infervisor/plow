/* wgmma_flash_prefill_probe.cu — standalone Hopper (sm_90a) WGMMA flash-attention PREFILL probe.
 *
 * GOAL: prove a FlashAttention-2 style prefill whose BOTH GEMMs are
 *   wgmma.mma_async.sync.aligned.m64nNk16.f32.bf16.bf16
 * (S = Q.K^T, then online softmax, then O += P.V) is numerically correct on real H100,
 * so op_attention.cuh's d_flash_prefill<HD,BQ,BKV> mma.sync mainloop can be replaced.
 *
 * NUMERICS CONTRACT (matches runtime/nvidia/op_attention.cuh):
 *   scale = 1/sqrt(HD); scores carried in LOG2 domain (lscale = scale*log2(e), exp -> ex2.approx),
 *   causal mask, online softmax (running max m, running sum l, accumulator rescale corr),
 *   GQA head grouping hkv = h / (n_head/n_kv_head).
 *
 * BUILD (GOTCHA: on CUDA 13.0 `-arch=sm_90a` silently emits compute_90 PTX and every wgmma is
 * rejected with "not supported on .target 'sm_90'". The explicit -gencode is required):
 *   env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin nvcc -std=c++17 \
 *       -gencode arch=compute_90a,code=sm_90a -O3 \
 *       -I runtime/common -I runtime/nvidia -include cstdint \
 *       runtime/nvidia/experiments/wgmma_flash_prefill_probe.cu -o <bin>
 *
 * MEASURED (H100 NVL): 12/12 cases PASS vs f32 CPU oracle, worst relL2 1.77e-3 (gate 3e-3);
 * HD=128 Bq=512 Tkv=4096 nh=8 -> 43.6 TFLOP/s vs 15.5 for the mma.sync baseline = 2.82x.
 */
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

typedef __nv_bfloat16 bf16;
#define NEG_BIG (-3.0e38f)

/* ============================ WGMMA plumbing ==================================================
 *
 * SHARED-MEMORY MATRIX DESCRIPTOR (64-bit), no-swizzle mode:
 *   bits [ 0,14) start address >> 4
 *   bits [16,30) leading-dim byte offset (LBO) >> 4
 *   bits [32,46) stride-dim byte offset (SBO) >> 4
 *   bits [49,52) base offset (0 here)
 *   bits [62,64) swizzle mode (0 = none)
 *
 * With swizzle=0 the operand is a grid of 8x8 bf16 "CORE MATRICES", and the hardware-fixed fact
 * that a plain row-major smem tile does NOT satisfy is: a core matrix is 128 CONTIGUOUS BYTES —
 * its 8 rows sit at a fixed 16-byte stride, NOT at the tile's row pitch. (Measured: with a plain
 * row-major tile the self-test reproduced C[0][0] exactly and nothing else, because only row 0
 * happens to land at the right address.) So every wgmma operand here is stored CORE-MATRIX
 * PACKED, K-inner (cm_off below), and then
 *   next core matrix along K  (c += 8) -> +128 bytes          => LBO = 128
 *   next core matrix along M/N(r += 8) -> +(W/8)*128 = 16*W B => SBO = 16*W
 * LBO is the K-direction stride and SBO the M/N-direction stride (also measured: the failing run
 * used LBO=16 and still contracted row 0's full K correctly).
 * All four operands are K-MAJOR (contraction contiguous), which lets one
 * code path serve both GEMMs (trans-a = trans-b = 0):
 *   D[M,N] = A[M,K] . B[N,K]^T          (A row-major MxK, B row-major NxK)
 *   GEMM0  S[64,BKV] = Q[64,HD]  . K[BKV,HD]^T      A = Qs,  B = Ks   (K natural, no transpose)
 *   GEMM1  O[64,HD]  = P[64,BKV] . V[BKV,HD]        A = Ps,  B = Vs   (MN-major, trans-b=1)
 * ...except V. P.V contracts over BKV, so a K-major B would have to be V^T[HD][BKV]; doing that
 * transpose in smem measured 46% of this kernel's runtime. Instead V is the ONE MN-major operand
 * (trans-b = 1): stored [K=BKV][N=HD], i.e. its natural layout, product = A.B with no implicit
 * transpose. MN-major SWAPS the descriptor stride roles -- LBO steps along K, SBO along N -- which
 * the trans-b=1 self-test below pins down (the non-swapped guess raises an illegal access).
 */
__device__ __forceinline__ uint64_t desc_enc(uint64_t x) { return (x & 0x3FFFFull) >> 4; }

__device__ __forceinline__ uint64_t make_desc_lbo_sbo(const void* ptr, uint32_t lbo,
                                                      uint32_t sbo) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    uint64_t d = 0;
    d |= desc_enc((uint64_t)addr);
    d |= desc_enc((uint64_t)lbo) << 16;
    d |= desc_enc((uint64_t)sbo) << 32;
    d |= (uint64_t)0 << 62; /* swizzle: none */
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc(const void* ptr, uint32_t row_w_elems) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    uint64_t d = 0;
    d |= desc_enc((uint64_t)addr);
    d |= desc_enc((uint64_t)128u) << 16;                /* LBO: next core matrix along K */
    d |= desc_enc((uint64_t)(16u * row_w_elems)) << 32; /* SBO: next core matrix along M/N */
    d |= (uint64_t)0 << 62;                             /* swizzle: none */
    return d;
}

/* Element offset (in bf16 ELEMENTS) of logical (r, c) inside a K-major core-matrix-packed tile of
 * logical shape [R][W]. Tile size is exactly R*W elements; 8 consecutive c (same c>>3) are still
 * 16 contiguous bytes, which is what keeps cp.async 16B copies legal straight into the tile. */
__device__ __forceinline__ int cm_off(int r, int c, int W) {
    return (r >> 3) * (W * 8) + (c >> 3) * 64 + (r & 7) * 8 + (c & 7);
}
/* Descriptor base for k-slice ks (columns [16*ks, 16*ks+16)) of such a tile: row 0, core col 2*ks. */
#define CM_KSLICE(ptr, ks) ((ptr) + (ks) * 128)

/* ACCUMULATOR FRAGMENT LAYOUT (f32, m64nNk16) — the thing that has to be right for softmax.
 * 128 threads = 4 warps; warp w owns rows [16w, 16w+16). Per 8-column block nb the 4 regs are the
 * classic m16n8 quad:
 *   reg[4*nb+0] -> (row = 16w + lane/4    , col = 8*nb + 2*(lane%4)    )
 *   reg[4*nb+1] -> (row = 16w + lane/4    , col = 8*nb + 2*(lane%4) + 1)
 *   reg[4*nb+2] -> (row = 16w + lane/4 + 8, col = 8*nb + 2*(lane%4)    )
 *   reg[4*nb+3] -> (row = 16w + lane/4 + 8, col = 8*nb + 2*(lane%4) + 1)
 * So a thread touches exactly TWO rows (rA = 16w+lane/4 and rB = rA+8) in BOTH accumulators —
 * which is what makes the per-row online-softmax rescale of the O accumulator a 2-value lookup. */
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

/* One warpgroup MMA: D[64][64] += A[64][16] . B[64][16]^T  (scale-d=1, scale-a=1, scale-b=1,
 * trans-a=0, trans-b=0).  n64 keeps the accumulator at 32 f32/lane, so the O accumulator is
 * software-tiled over HD/64 of these instead of one 128-register m64n128 monster. */
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

/* trans-b = 1 twin: B is MN-major, i.e. stored [K][N] with N contiguous, and the product is
 * D = A . B (no implicit transpose of B). This is what lets V feed P.V in its NATURAL [kv][hd]
 * layout, killing the smem transpose. Which of LBO/SBO then steps along K is the open question the
 * tb1 self-test below settles. */
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

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem) : "memory");
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}
__device__ __forceinline__ void cp_async_wait0() {
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

__device__ __forceinline__ float ex2(float x) {
    float r;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

/* ============================ PHASE A: wgmma layout self-test =================================
 * C[64][64] = A[64][SELF_K] . B[64][SELF_K]^T, both operands plain row-major in smem.
 * Locks the descriptor (LBO/SBO/swizzle), the trans flags, and the accumulator (row,col) map
 * before any flash logic runs. */
#define SELF_K 64
__global__ void k_wgmma_selftest(const bf16* __restrict__ A, const bf16* __restrict__ B,
                                 float* __restrict__ C) {
    extern __shared__ char smem_raw[];
    bf16* As = (bf16*)smem_raw;  /* [64][SELF_K] */
    bf16* Bs = As + 64 * SELF_K; /* [64][SELF_K] */
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;

    for (int i = tid; i < 64 * SELF_K; i += 128) {
        const int r = i / SELF_K, c = i % SELF_K;
        As[cm_off(r, c, SELF_K)] = A[i];
        Bs[cm_off(r, c, SELF_K)] = B[i];
    }
    __syncthreads();
    async_proxy_fence();

    float d[32];
#pragma unroll
    for (int i = 0; i < 32; i++) d[i] = 0.0f;

    wgmma_fence();
#pragma unroll 1
    for (int ks = 0; ks < SELF_K / 16; ks++)
        wgmma_m64n64k16(d, make_smem_desc(CM_KSLICE(As, ks), SELF_K),
                        make_smem_desc(CM_KSLICE(Bs, ks), SELF_K));
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

/* trans-b=1 layout probe: C[64][64] = A[64][SELF_K] . B2[SELF_K][64], B2 stored [K][N] packed.
 * VARIANT 0: LBO=128 (steps along N), SBO=16*N (steps along K row-blocks)  -- same form as K-major
 * VARIANT 1: LBO/SBO swapped. */
template <int VARIANT>
__global__ void k_wgmma_selftest_tb1(const bf16* __restrict__ A, const bf16* __restrict__ B2,
                                     float* __restrict__ C) {
    extern __shared__ char smem_raw[];
    bf16* As = (bf16*)smem_raw;  /* [64][SELF_K] packed, W=SELF_K */
    bf16* Bs = As + 64 * SELF_K; /* [SELF_K][64] packed, W=64     */
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    for (int i = tid; i < 64 * SELF_K; i += 128) {
        As[cm_off(i / SELF_K, i % SELF_K, SELF_K)] = A[i];
        Bs[cm_off(i / 64, i % 64, 64)] = B2[i];
    }
    __syncthreads();
    async_proxy_fence();
    float d[32];
#pragma unroll
    for (int i = 0; i < 32; i++) d[i] = 0.0f;
    wgmma_fence();
#pragma unroll 1
    for (int ks = 0; ks < SELF_K / 16; ks++) {
        /* B k-slice = rows [16ks,16ks+16) of a [K][64] packed tile -> +2*ks row-blocks */
        const bf16* bp = Bs + ks * 16 * 64;
        const uint64_t db = (VARIANT == 0) ? make_desc_lbo_sbo(bp, 128, 16 * 64)
                                           : make_desc_lbo_sbo(bp, 16 * 64, 128);
        wgmma_m64n64k16_tb1(d, make_smem_desc(CM_KSLICE(As, ks), SELF_K), db);
    }
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

/* ============================ PHASE B: WGMMA flash prefill ====================================
 * One warpgroup (128 threads) per (head, q-tile). BQ = 64 = the wgmma m64.
 *
 * Q  : [n_head][seq_q][HD]      bf16   (per-head contiguous)
 * K,V: [n_kv_head][seq_kv][HD]  bf16
 * O  : [n_head][seq_q][HD]      f32
 *
 * Causal: query row i of the tile sits at absolute position q_pos0 + i with
 * q_pos0 = seq_kv - seq_q + qt*BQ (the prefill chunk is the tail of the sequence), so row i
 * attends kv j iff j <= q_pos0 + i.
 *
 * ONLINE SOFTMAX / ACCUMULATOR CONSISTENCY:
 *   - GEMM0's f32 accumulator is scaled + causally masked IN THE FRAGMENT (each lane knows its
 *     (row,col) from the ACC_ROW_A / ACC_ROW_B / ACC_COL macros) and spilled to Ss[BQ][BKV] f32.
 *   - Row r's whole online state (m, l) lives in the registers of ONE thread (tid == r, r < BQ),
 *     which reads the full Ss row, updates (m, l), writes P into Ps[BQ][BKV] bf16 (the GEMM1 A
 *     operand, K-major in BKV) and publishes corr[r].
 *   - Every thread then rescales its O-accumulator regs by corr[rA] / corr[rB]. Because a thread
 *     owns exactly two rows in the accumulator, that is two smem reads, and it is by construction
 *     the SAME (row,col) map GEMM0 used — no re-derivation, no drift.
 */
template <int HD, int BQ, int BKV>
__global__ __launch_bounds__(128) void k_wgmma_flash_prefill(
    const bf16* __restrict__ Q, const bf16* __restrict__ K, const bf16* __restrict__ V,
    float* __restrict__ O, int seq_q, int seq_kv, int n_head, int n_kv_head, float lscale) {
    static_assert(BQ == 64, "one warpgroup m64");
    static_assert(HD % 64 == 0, "O accumulator tiles by n64");
    static_assert(BKV % 16 == 0, "k16 steps");
    constexpr int NT = HD / 64;   /* O accumulator n64 tiles */
    constexpr int KS0 = HD / 16;  /* GEMM0 k16 steps         */
    constexpr int KS1 = BKV / 16; /* GEMM1 k16 steps         */

    extern __shared__ char smem_raw[];
    bf16* Qs = (bf16*)smem_raw;          /* [BQ ][HD ] */
    bf16* Ks = Qs + BQ * HD;             /* [BKV][HD ] */
    bf16* Vs = Ks + BKV * HD;            /* [BKV][HD ] natural = GEMM1 B, MN-major */
    bf16* Ps = Vs + BKV * HD;            /* [BQ ][BKV] GEMM1 A                     */

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int h = blockIdx.x;
    const int qt = blockIdx.y;
    const int gqa = n_head / n_kv_head;
    const int hkv = h / gqa;
    const int q_pos0 = seq_kv - seq_q + qt * BQ;

    const bf16* Qh = Q + ((size_t)h * seq_q + (size_t)qt * BQ) * HD;
    const bf16* Kh = K + (size_t)hkv * seq_kv * HD;
    const bf16* Vh = V + (size_t)hkv * seq_kv * HD;

    const int qrows = (seq_q - qt * BQ < BQ) ? (seq_q - qt * BQ) : BQ;

    /* --- stage Q once (cp.async straight into the core-matrix-packed tile; 8 consecutive HD
     * elements are still 16 contiguous bytes there. Rows beyond seq_q zero-filled) --- */
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

    /* Online state lives in REGISTERS, two rows per lane — exactly the two accumulator rows this
     * lane owns (rA, rB). It is replicated across each quad {4q..4q+3} (those 4 lanes share the
     * same two rows and hold disjoint column sets), and the butterfly shuffles below keep the four
     * copies bit-identical, so no smem and no broadcast barrier is needed. */
    const int rA = ACC_ROW_A(warp, lane), rB = ACC_ROW_B(warp, lane);
    const int qposA = q_pos0 + rA, qposB = q_pos0 + rB;
    float mA = NEG_BIG, lA = 0.0f, mB = NEG_BIG, lB = 0.0f;

    const int ntiles = (seq_kv + BKV - 1) / BKV;
    for (int t = 0; t < ntiles; t++) {
        const int kvbase = t * BKV;
        const int krows = (seq_kv - kvbase < BKV) ? (seq_kv - kvbase) : BKV;

        /* --- cp.async K and V, both core-matrix packed [BKV][HD] (both ARE wgmma operands: K is
         * the K-major B of QK^T, V is the MN-major B of P.V). Tail rows zero-filled, so a masked
         * P=0 never multiplies stale data. NO V TRANSPOSE: the trans-b=1 descriptor consumes V in
         * its natural layout, which measured 46% of this kernel's runtime when done by hand. --- */
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

        /* --- GEMM0: S[BQ][BKV] = Q . K^T --- */
        float S[32];
#pragma unroll
        for (int i = 0; i < 32; i++) S[i] = 0.0f;
        wgmma_fence();
#pragma unroll 1
        for (int ks = 0; ks < KS0; ks++)
            wgmma_m64n64k16(S, make_smem_desc(CM_KSLICE(Qs, ks), HD),
                            make_smem_desc(CM_KSLICE(Ks, ks), HD));
        wgmma_commit();
        wgmma_wait0();

        /* --- scale + causal mask IN THE FRAGMENT (no smem round trip) --- */
        float mxA = NEG_BIG, mxB = NEG_BIG;
#pragma unroll
        for (int nb = 0; nb < BKV / 8; nb++) {
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
        }
        /* Row reduction = QUAD butterfly. A row's BKV columns are split over the 4 lanes that
         * share (lane>>2); xor-1 and xor-2 stay inside that quad, so two shuffles complete the
         * full row max / row sum and leave all 4 lanes with identical values. */
        mxA = fmaxf(mxA, __shfl_xor_sync(0xffffffffu, mxA, 1));
        mxA = fmaxf(mxA, __shfl_xor_sync(0xffffffffu, mxA, 2));
        mxB = fmaxf(mxB, __shfl_xor_sync(0xffffffffu, mxB, 1));
        mxB = fmaxf(mxB, __shfl_xor_sync(0xffffffffu, mxB, 2));

        const float mnA = fmaxf(mA, mxA), mnB = fmaxf(mB, mxB);
        const float cA = ex2(mA - mnA), cB = ex2(mB - mnB);
        mA = mnA;
        mB = mnB;

        /* P -> the GEMM1 A operand. The two e-halves are adjacent columns of the SAME core matrix
         * (col&7 = 2*(lane&3)+e), so each pair is one 32-bit bf16x2 store. */
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

        /* --- rescale the O accumulator on the SAME two rows this lane owns --- */
#pragma unroll
        for (int nt = 0; nt < NT; nt++)
#pragma unroll
            for (int nb = 0; nb < 8; nb++)
#pragma unroll
                for (int e = 0; e < 2; e++) {
                    Oacc[nt][4 * nb + e] *= cA;
                    Oacc[nt][4 * nb + 2 + e] *= cB;
                }
        __syncthreads(); /* Ps complete for the whole warpgroup */
        async_proxy_fence();

        /* --- GEMM1: O += P . V.  A = Ps (K-major in BKV), B = Vs (MN-MAJOR, trans-b=1):
         * B's k-slice ks = rows [16ks,16ks+16) of the packed [BKV][HD] tile -> +ks*16*HD elements;
         * B's n-tile nt = HD columns [64nt, 64nt+64) -> +nt*512 elements (8 core cols of 64).
         * MN-major swaps the stride roles: LBO steps along K (row-block = 16*HD B), SBO along N
         * (one core matrix = 128 B). --- */
        wgmma_fence();
#pragma unroll
        for (int nt = 0; nt < NT; nt++)
#pragma unroll 1
            for (int ks = 0; ks < KS1; ks++)
                wgmma_m64n64k16_tb1(
                    Oacc[nt], make_smem_desc(CM_KSLICE(Ps, ks), BKV),
                    make_desc_lbo_sbo(Vs + ks * 16 * HD + nt * 512, 16 * HD, 128));
        wgmma_commit();
        wgmma_wait0();
        __syncthreads(); /* Ss/Ps/Ks/Vs reused next tile */
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

/* ============================ mma.sync m16n8k16 BASELINE ======================================
 * Same tiling / same numerics, warp-scoped tensor cores (the atom op_attention.cuh uses today), so
 * the wgmma speedup is measured against plow's own current prefill math, not a strawman.
 * 8 warps (256 thr): QK^T = (BQ/16) query-warps x WN kv-warps; P.V = same grid over HD. */
__device__ __forceinline__ void b_ldmatrix_x4(unsigned (&r)[4], const void* s) {
    unsigned a = (unsigned)__cvta_generic_to_shared(s);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void b_ldmatrix_x2(unsigned (&r)[2], const void* s) {
    unsigned a = (unsigned)__cvta_generic_to_shared(s);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}
__device__ __forceinline__ void b_ldmatrix_x2_trans(unsigned (&r)[2], const void* s) {
    unsigned a = (unsigned)__cvta_generic_to_shared(s);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}
__device__ __forceinline__ void b_mma(float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2],
                                      const float (&c)[4]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]),
                   "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

#define BPAD 8
template <int HD, int BQ, int BKV>
__global__ __launch_bounds__(256) void k_mma_flash_prefill(
    const bf16* __restrict__ Q, const bf16* __restrict__ K, const bf16* __restrict__ V,
    float* __restrict__ O, int seq_q, int seq_kv, int n_head, int n_kv_head, float lscale) {
    extern __shared__ char smem_raw[];
    bf16* Qs = (bf16*)smem_raw;                   /* [BQ ][HD+BPAD ] */
    bf16* Ks = Qs + BQ * (HD + BPAD);             /* [BKV][HD+BPAD ] */
    bf16* Vs = Ks + BKV * (HD + BPAD);            /* [BKV][HD+BPAD ] */
    bf16* Ps = Vs + BKV * (HD + BPAD);            /* [BQ ][BKV+BPAD] */
    float* Ss = (float*)(Ps + BQ * (BKV + BPAD)); /* [BQ ][BKV] */
    float* corr_a = Ss + BQ * BKV;
    float* l_a = corr_a + BQ;

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int h = blockIdx.x, qt = blockIdx.y;
    const int gqa = n_head / n_kv_head, hkv = h / gqa;
    const int q_pos0 = seq_kv - seq_q + qt * BQ;
    const bf16* Qh = Q + ((size_t)h * seq_q + (size_t)qt * BQ) * HD;
    const bf16* Kh = K + (size_t)hkv * seq_kv * HD;
    const bf16* Vh = V + (size_t)hkv * seq_kv * HD;
    const int qrows = (seq_q - qt * BQ < BQ) ? (seq_q - qt * BQ) : BQ;

    for (int i = tid * 8; i < BQ * HD; i += 256 * 8) {
        const int r = i / HD, c = i % HD;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (r < qrows) v = *(const uint4*)(Qh + (size_t)r * HD + c);
        *(uint4*)(Qs + r * (HD + BPAD) + c) = v;
    }

    constexpr int WQ = BQ / 16;        /* query warps */
    constexpr int WN = 256 / 32 / WQ;  /* kv / hd warps */
    const int wq = warp / WN, wn = warp % WN;
    constexpr int NKV = BKV / WN; /* kv cols per warp in QK  */
    constexpr int NHD = HD / WN;  /* hd cols per warp in P.V */

    float oacc[NHD / 8][4];
#pragma unroll
    for (int j = 0; j < NHD / 8; j++)
#pragma unroll
        for (int i = 0; i < 4; i++) oacc[j][i] = 0.0f;
    float m_r = NEG_BIG, l_r = 0.0f;

    const int ntiles = (seq_kv + BKV - 1) / BKV;
    __syncthreads();
    for (int t = 0; t < ntiles; t++) {
        const int kvbase = t * BKV;
        const int krows = (seq_kv - kvbase < BKV) ? (seq_kv - kvbase) : BKV;
        for (int i = tid * 8; i < BKV * HD; i += 256 * 8) {
            const int r = i / HD, c = i % HD;
            uint4 kv4 = make_uint4(0, 0, 0, 0), vv4 = make_uint4(0, 0, 0, 0);
            if (r < krows) {
                kv4 = *(const uint4*)(Kh + (size_t)(kvbase + r) * HD + c);
                vv4 = *(const uint4*)(Vh + (size_t)(kvbase + r) * HD + c);
            }
            *(uint4*)(Ks + r * (HD + BPAD) + c) = kv4;
            *(uint4*)(Vs + r * (HD + BPAD) + c) = vv4;
        }
        __syncthreads();

        /* QK^T: warp (wq,wn) owns rows [16*wq,+16) x cols [NKV*wn,+NKV) */
        float sacc[NKV / 8][4];
#pragma unroll
        for (int j = 0; j < NKV / 8; j++)
#pragma unroll
            for (int i = 0; i < 4; i++) sacc[j][i] = 0.0f;
#pragma unroll 1
        for (int k = 0; k < HD; k += 16) {
            unsigned a[4];
            b_ldmatrix_x4(a, Qs + (16 * wq + (lane & 15)) * (HD + BPAD) + k + ((lane >> 4) << 3));
#pragma unroll
            for (int j = 0; j < NKV / 8; j++) {
                unsigned b[2];
                b_ldmatrix_x2(b, Ks + (NKV * wn + 8 * j + (lane & 7)) * (HD + BPAD) + k +
                                     ((lane >> 3) & 3) * 8);
                float o[4] = {sacc[j][0], sacc[j][1], sacc[j][2], sacc[j][3]};
                b_mma(sacc[j], a, b, o);
            }
        }
#pragma unroll
        for (int j = 0; j < NKV / 8; j++) {
            const int r0 = 16 * wq + (lane >> 2), c0 = NKV * wn + 8 * j + 2 * (lane & 3);
#pragma unroll
            for (int e = 0; e < 2; e++) {
                const int kvp0 = kvbase + c0 + e;
                Ss[r0 * BKV + c0 + e] =
                    (kvp0 < seq_kv && kvp0 <= q_pos0 + r0) ? sacc[j][e] * lscale : NEG_BIG;
                Ss[(r0 + 8) * BKV + c0 + e] =
                    (kvp0 < seq_kv && kvp0 <= q_pos0 + r0 + 8) ? sacc[j][2 + e] * lscale : NEG_BIG;
            }
        }
        __syncthreads();
        if (tid < BQ) {
            const float* srow = Ss + tid * BKV;
            float mx = NEG_BIG;
            for (int j = 0; j < BKV; j++) mx = fmaxf(mx, srow[j]);
            const float mnew = fmaxf(m_r, mx);
            const float corr = ex2(m_r - mnew);
            float sum = 0.0f;
            for (int j = 0; j < BKV; j++) {
                const float p = ex2(srow[j] - mnew);
                sum += p;
                Ps[tid * (BKV + BPAD) + j] = __float2bfloat16(p);
            }
            l_r = l_r * corr + sum;
            m_r = mnew;
            corr_a[tid] = corr;
        }
        __syncthreads();

        /* P.V: warp (wq,wn) owns rows [16*wq,+16) x hd [NHD*wn,+NHD) */
        {
            const int r0 = 16 * wq + (lane >> 2);
            const float cA = corr_a[r0], cB = corr_a[r0 + 8];
#pragma unroll
            for (int j = 0; j < NHD / 8; j++) {
                oacc[j][0] *= cA;
                oacc[j][1] *= cA;
                oacc[j][2] *= cB;
                oacc[j][3] *= cB;
            }
        }
#pragma unroll 1
        for (int k = 0; k < BKV; k += 16) {
            unsigned a[4];
            b_ldmatrix_x4(a, Ps + (16 * wq + (lane & 15)) * (BKV + BPAD) + k + ((lane >> 4) << 3));
#pragma unroll
            for (int j = 0; j < NHD / 8; j++) {
                unsigned b[2];
                b_ldmatrix_x2_trans(b, Vs + (k + (lane & 7) + ((lane >> 3) & 1) * 8) * (HD + BPAD) +
                                           NHD * wn + 8 * j);
                float o[4] = {oacc[j][0], oacc[j][1], oacc[j][2], oacc[j][3]};
                b_mma(oacc[j], a, b, o);
            }
        }
        __syncthreads();
    }
    if (tid < BQ) l_a[tid] = l_r;
    __syncthreads();
    {
        const int r0 = 16 * wq + (lane >> 2);
        const float lA = l_a[r0], lB = l_a[r0 + 8];
        const float iA = lA > 0.f ? 1.f / lA : 0.f, iB = lB > 0.f ? 1.f / lB : 0.f;
        float* Oh = O + ((size_t)h * seq_q + (size_t)qt * BQ) * HD;
#pragma unroll
        for (int j = 0; j < NHD / 8; j++) {
            const int c0 = NHD * wn + 8 * j + 2 * (lane & 3);
#pragma unroll
            for (int e = 0; e < 2; e++) {
                if (r0 < qrows) Oh[(size_t)r0 * HD + c0 + e] = oacc[j][e] * iA;
                if (r0 + 8 < qrows) Oh[(size_t)(r0 + 8) * HD + c0 + e] = oacc[j][2 + e] * iB;
            }
        }
    }
}

/* ============================ host: oracle, harness ============================================ */
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

static int run_selftest() {
    std::vector<bf16> A(64 * SELF_K), B(64 * SELF_K);
    for (int i = 0; i < 64 * SELF_K; i++) {
        A[i] = f2bf(frand());
        B[i] = f2bf(frand());
    }
    std::vector<float> ref(64 * 64, 0.f);
    for (int i = 0; i < 64; i++)
        for (int j = 0; j < 64; j++) {
            float d = 0;
            for (int k = 0; k < SELF_K; k++) d += bf2f(A[i * SELF_K + k]) * bf2f(B[j * SELF_K + k]);
            ref[i * 64 + j] = d;
        }
    bf16 *dA, *dB;
    float* dC;
    CK(cudaMalloc(&dA, A.size() * 2));
    CK(cudaMalloc(&dB, B.size() * 2));
    CK(cudaMalloc(&dC, 64 * 64 * 4));
    CK(cudaMemcpy(dA, A.data(), A.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, B.data(), B.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemset(dC, 0, 64 * 64 * 4));
    const size_t sm = 2 * 64 * SELF_K * sizeof(bf16);
    k_wgmma_selftest<<<1, 128, sm>>>(dA, dB, dC);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    std::vector<float> got(64 * 64);
    CK(cudaMemcpy(got.data(), dC, 64 * 64 * 4, cudaMemcpyDeviceToHost));
    const double e = rel_l2(got, ref);
    printf("[selftest] wgmma m64n64k16 (K=%d) A.B^T relL2 = %.3e  %s\n", SELF_K, e,
           e < 3e-3 ? "PASS" : "FAIL");
    if (e >= 3e-3) {
        printf("   ref r0 %.5f %.5f %.5f %.5f | r1 %.5f %.5f\n", ref[0], ref[1], ref[2], ref[3],
               ref[64], ref[65]);
        printf("   got r0 %.5f %.5f %.5f %.5f | r1 %.5f %.5f\n", got[0], got[1], got[2], got[3],
               got[64], got[65]);
    }
    CK(cudaFree(dA));
    CK(cudaFree(dB));
    CK(cudaFree(dC));
    return e < 3e-3;
}

static void run_selftest_tb1() {
    std::vector<bf16> A(64 * SELF_K), B2(SELF_K * 64);
    for (auto& x : A) x = f2bf(frand());
    for (auto& x : B2) x = f2bf(frand());
    std::vector<float> ref(64 * 64, 0.f);
    for (int i = 0; i < 64; i++)
        for (int j = 0; j < 64; j++) {
            float d = 0;
            for (int k = 0; k < SELF_K; k++) d += bf2f(A[i * SELF_K + k]) * bf2f(B2[k * 64 + j]);
            ref[i * 64 + j] = d;
        }
    bf16 *dA, *dB;
    float* dC;
    CK(cudaMalloc(&dA, A.size() * 2));
    CK(cudaMalloc(&dB, B2.size() * 2));
    CK(cudaMalloc(&dC, 64 * 64 * 4));
    CK(cudaMemcpy(dA, A.data(), A.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, B2.data(), B2.size() * 2, cudaMemcpyHostToDevice));
    const size_t sm = 2 * 64 * SELF_K * sizeof(bf16);
    std::vector<float> got(64 * 64);
    /* VARIANT 0 walks off the end of the tile and raises an illegal access (it steps the N
     * direction by SBO=16N), which is itself the evidence that for an MN-major operand the roles
     * are SWAPPED vs K-major: LBO steps along K (row-blocks), SBO along N. Only variant 1 is run;
     * an illegal access is sticky and would kill the context for every later case. */
    for (int v = 1; v < 2; v++) {
        CK(cudaMemset(dC, 0, 64 * 64 * 4));
        if (v == 0) k_wgmma_selftest_tb1<0><<<1, 128, sm>>>(dA, dB, dC);
        else k_wgmma_selftest_tb1<1><<<1, 128, sm>>>(dA, dB, dC);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(got.data(), dC, 64 * 64 * 4, cudaMemcpyDeviceToHost));
        const double e = rel_l2(got, ref);
        printf("[selftest] trans-b=1 (MN-major B) variant %d (%s) relL2 = %.3e  %s\n", v,
               v == 0 ? "LBO=128,SBO=16N" : "LBO=16N,SBO=128", e, e < 3e-3 ? "PASS" : "fail");
    }
    CK(cudaFree(dA));
    CK(cudaFree(dB));
    CK(cudaFree(dC));
}

template <int HD, int BQ, int BKV> static size_t wg_smem() {
    return (size_t)(BQ * HD + BKV * HD + BKV * HD + BQ * BKV) * sizeof(bf16);
}
template <int HD, int BQ, int BKV> static size_t mm_smem() {
    return (size_t)(BQ * (HD + BPAD) + 2 * BKV * (HD + BPAD) + BQ * (BKV + BPAD)) * sizeof(bf16) +
           (size_t)(BQ * BKV + BQ + BQ) * sizeof(float);
}

struct Case {
    int seq_q, seq_kv, nh, nkv;
};

template <int HD, int BQ, int BKV>
static int run_case(const Case& c, int& ncase, double& worst) {
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
    CK(cudaMemset(dO, 0, (size_t)nh * seq_q * HD * 4));

    const float lscale = (1.0f / sqrtf((float)HD)) * 1.4426950408889634f;
    const size_t sm = wg_smem<HD, BQ, BKV>();
    CK(cudaFuncSetAttribute(k_wgmma_flash_prefill<HD, BQ, BKV>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm));
    dim3 grid(nh, (seq_q + BQ - 1) / BQ);
    k_wgmma_flash_prefill<HD, BQ, BKV>
        <<<grid, 128, sm>>>(dQ, dK, dV, dO, seq_q, seq_kv, nh, nkv, lscale);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    std::vector<float> got((size_t)nh * seq_q * HD);
    CK(cudaMemcpy(got.data(), dO, got.size() * 4, cudaMemcpyDeviceToHost));
    const double e = rel_l2(got, ref);
    printf("[case] HD=%3d Bq=%3d Tkv=%-5d nh=%d nkv=%d (gqa=%d)  relL2 = %.3e  %s\n", HD, seq_q,
           seq_kv, nh, nkv, nh / nkv, e, e < 3e-3 ? "PASS" : "FAIL");
    if (e >= 3e-3 && ncase == 0) {
        printf("   ref[0..5] %.5f %.5f %.5f %.5f %.5f %.5f\n", ref[0], ref[1], ref[2], ref[3],
               ref[4], ref[5]);
        printf("   got[0..5] %.5f %.5f %.5f %.5f %.5f %.5f\n", got[0], got[1], got[2], got[3],
               got[4], got[5]);
    }
    ncase++;
    if (e > worst) worst = e;
    CK(cudaFree(dQ));
    CK(cudaFree(dK));
    CK(cudaFree(dV));
    CK(cudaFree(dO));
    return e < 3e-3;
}

template <int HD, int BQ, int BKV> static int run_case_mma(const Case& c) {
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
    CK(cudaMemset(dO, 0, (size_t)nh * seq_q * HD * 4));
    const float lscale = (1.0f / sqrtf((float)HD)) * 1.4426950408889634f;
    const size_t sm = mm_smem<HD, BQ, BKV>();
    CK(cudaFuncSetAttribute(k_mma_flash_prefill<HD, BQ, BKV>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm));
    dim3 grid(nh, (seq_q + BQ - 1) / BQ);
    k_mma_flash_prefill<HD, BQ, BKV>
        <<<grid, 256, sm>>>(dQ, dK, dV, dO, seq_q, seq_kv, nh, nkv, lscale);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    std::vector<float> got((size_t)nh * seq_q * HD);
    CK(cudaMemcpy(got.data(), dO, got.size() * 4, cudaMemcpyDeviceToHost));
    const double e = rel_l2(got, ref);
    printf("[base] mma.sync HD=%d Bq=%d Tkv=%-5d nh=%d nkv=%d  relL2 = %.3e  %s\n", HD, seq_q,
           seq_kv, nh, nkv, e, e < 3e-3 ? "PASS" : "FAIL");
    CK(cudaFree(dQ));
    CK(cudaFree(dK));
    CK(cudaFree(dV));
    CK(cudaFree(dO));
    return e < 3e-3;
}

template <int HD, int BQ, int BKV>
static void bench(int seq_q, int seq_kv, int nh, int nkv, int iters, int warm) {
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
    const float lscale = (1.0f / sqrtf((float)HD)) * 1.4426950408889634f;
    dim3 grid(nh, (seq_q + BQ - 1) / BQ);

    const size_t swg = wg_smem<HD, BQ, BKV>();
    CK(cudaFuncSetAttribute(k_wgmma_flash_prefill<HD, BQ, BKV>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)swg));
    const size_t smm = mm_smem<HD, BQ, BKV>();
    CK(cudaFuncSetAttribute(k_mma_flash_prefill<HD, BQ, BKV>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smm));

    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    float t_wg = 0, t_mm = 0;

    for (int i = 0; i < warm; i++)
        k_wgmma_flash_prefill<HD, BQ, BKV>
            <<<grid, 128, swg>>>(dQ, dK, dV, dO, seq_q, seq_kv, nh, nkv, lscale);
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(a));
    for (int i = 0; i < iters; i++)
        k_wgmma_flash_prefill<HD, BQ, BKV>
            <<<grid, 128, swg>>>(dQ, dK, dV, dO, seq_q, seq_kv, nh, nkv, lscale);
    CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    CK(cudaEventElapsedTime(&t_wg, a, b));

    for (int i = 0; i < warm; i++)
        k_mma_flash_prefill<HD, BQ, BKV>
            <<<grid, 256, smm>>>(dQ, dK, dV, dO, seq_q, seq_kv, nh, nkv, lscale);
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(a));
    for (int i = 0; i < iters; i++)
        k_mma_flash_prefill<HD, BQ, BKV>
            <<<grid, 256, smm>>>(dQ, dK, dV, dO, seq_q, seq_kv, nh, nkv, lscale);
    CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    CK(cudaEventElapsedTime(&t_mm, a, b));

    const double flop = 4.0 * (double)seq_q * seq_kv * HD * nh;
    const double ms_wg = t_wg / iters, ms_mm = t_mm / iters;
    printf("[bench] HD=%d Bq=%d Tkv=%d nh=%d nkv=%d  iters=%d warm=%d\n", HD, seq_q, seq_kv, nh,
           nkv, iters, warm);
    printf("        wgmma    %8.4f ms  %8.2f TFLOP/s   (smem %zu B, 128 thr)\n", ms_wg,
           flop / (ms_wg * 1e-3) / 1e12, swg);
    printf("        mma.sync %8.4f ms  %8.2f TFLOP/s   (smem %zu B, 256 thr)\n", ms_mm,
           flop / (ms_mm * 1e-3) / 1e12, smm);
    printf("        speedup  %.3fx\n", ms_mm / ms_wg);
    CK(cudaEventDestroy(a));
    CK(cudaEventDestroy(b));
    CK(cudaFree(dQ));
    CK(cudaFree(dK));
    CK(cudaFree(dV));
    CK(cudaFree(dO));
}

int main(int argc, char** argv) {
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    printf("GPU: %s  sm_%d%d  smem/block(optin) %zu\n", p.name, p.major, p.minor,
           (size_t)p.sharedMemPerBlockOptin);
    if (p.major != 9) {
        printf("RESULT: SKIP (needs sm_90)\n");
        return 0;
    }

    if (!run_selftest()) {
        printf("RESULT: FAIL (wgmma layout self-test)\n");
        return 1;
    }
    run_selftest_tb1();

    int ok = 1, ncase = 0;
    double worst = 0;
    ok &= run_case<128, 64, 64>({64, 64, 4, 1}, ncase, worst);
    ok &= run_case<128, 64, 64>({64, 256, 4, 1}, ncase, worst);
    ok &= run_case<128, 64, 64>({64, 1024, 4, 1}, ncase, worst);
    ok &= run_case<128, 64, 64>({64, 64, 4, 2}, ncase, worst);
    ok &= run_case<128, 64, 64>({64, 256, 4, 2}, ncase, worst);
    ok &= run_case<128, 64, 64>({64, 1024, 4, 2}, ncase, worst);
    ok &= run_case<128, 64, 64>({64, 200, 4, 1}, ncase, worst); /* non-multiple Tkv */
    ok &= run_case<128, 64, 64>({64, 200, 4, 2}, ncase, worst);
    ok &= run_case<128, 64, 64>({128, 1024, 4, 2}, ncase, worst); /* multi q-tile */
    ok &= run_case<256, 64, 64>({64, 256, 4, 1}, ncase, worst);
    ok &= run_case<256, 64, 64>({64, 1024, 4, 2}, ncase, worst);
    ok &= run_case<256, 64, 64>({64, 200, 4, 1}, ncase, worst);

    printf("RESULT: %s  (%d cases, worst relL2 = %.3e, gate 3e-3)\n", ok ? "PASS" : "FAIL", ncase,
           worst);
    if (!ok) return 1;

    if (!run_case_mma<128, 64, 64>({64, 1024, 4, 1}))
        printf("WARN: mma.sync baseline failed its oracle; speedup number is unsafe\n");

    if (argc < 2 || strcmp(argv[1], "--noperf") != 0) bench<128, 64, 64>(512, 4096, 8, 1, 100, 10);
    return 0;
}
