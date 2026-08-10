// h100 sm_90a WGMMA bf16 GEMM microkernel — correctness-first probe + benchmark.
//
// Contract (plow prefill linear, "TN" GEMM, BOTH operands K-contiguous):
//   C[m,n] = sum_k A[m,k] * B[n,k]         ==  C = A . B^T
//   A: bf16 row-major [M,K] (k contiguous)
//   B: bf16 row-major [N,K] (k contiguous, row n = output channel n)
//   C: f32 [M,N]
// Exactly what op_gemm.cuh does today with mma.sync.aligned.m16n8k16.row.col
// (pgm_mma, B stored [n][k]); this file is the WGMMA (warpgroup) equivalent,
// plus a structurally identical mma.sync baseline for a fair speedup number.
//
// WGMMA shape: m64n128k16 (one warpgroup = 128 threads per block).
//   N=128 over N=256: 64 f32 accumulators/thread instead of 128, so the K
//   pipeline + descriptors fit without spilling; 128 divides every Gemma
//   prefill N (4096, 15360). Still a full warpgroup-wide wgmma issue.
//
// smem layout: 128-BYTE SWIZZLE, K-major, trans_a = trans_b = 0.
//   LOGICAL layout is plain row-major [rows][BK=64] (row = 64 bf16 = 128 B = one
//   swizzle line = 8 chunks of 16 B). PHYSICAL layout XORs the chunk index with
//   the row-within-group:
//       phys_elem(row, c) = row*64 + ((c ^ (row & 7)) * 8),  c = k/8 in 0..7
//   Descriptor: LBO = 16 B (logical k-core stride), SBO = 1024 B (8 rows x 128 B),
//   swizzle-mode bits[63:62] = 1 (128 B). The tile base is rounded to 1024 B so
//   the hardware's address-bit swizzle ([6:4] chunk ^ [9:7] row) lines up with
//   ours and the matrix-base-offset field stays 0. Advancing one k16 wgmma
//   substep = +2 chunks = +32 B = +16 bf16 on the descriptor start address only.
//
//   (A no-swizzle core-matrix-blocked layout was implemented first and is also
//   numerically correct -- LBO=128, SBO=(BK/8)*128, descriptor step +256 B --
//   but its 1024 B row-core stride puts every row-core on the same smem bank:
//   8-way conflicts on both the cp.async store and the wgmma operand read cost
//   ~2x. The 128 B swizzle is what makes it fast, not what makes it correct.)
//
// Build (compilation needs NO gpu lock; every RUN does):
//   env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin nvcc -std=c++17 \
//     -gencode arch=compute_90a,code=sm_90a -O3 -I runtime/common -I runtime/nvidia \
//     -include cstdint runtime/nvidia/experiments/wgmma_bf16_probe.cu -o BUILD/wgmma_bf16_probe
//   flock /tmp/plow_gpu.lock BUILD/wgmma_bf16_probe
// NOTE: `-arch=sm_90a` alone did NOT forward the `a` (arch-accelerated) feature
// set to ptxas with CUDA 13.0 -> "wgmma.fence not supported on .target sm_90".
// The explicit `-gencode arch=compute_90a,code=sm_90a` form is required.
//
// Measured on H100 NVL (132 SM), 100 iters + 10 warmup, all configs relL2-PASS.
// -DWGMMA_MS = m64 slabs per warpgroup (BM = 64*MS), -DWGMMA_NS = cp.async depth.
//   MS NS | (512,4096,3840)        | (512,15360,3840)
//    1  3 | 177 TF/s  (1.28x mma)  | 161 TF/s  (1.70x mma)   <- default
//    1  4 | 179 TF/s  (1.21x)      | 153 TF/s  (1.40x)
//    2  2 | 133 TF/s  (1.51x)      | 184 TF/s  (1.52x)
//    2  3 | 156 TF/s  (1.48x)      | 219 TF/s  (1.55x)
// MS=1 is the default because MS=2 (BM=128) leaves only 4*32 = 128 output tiles
// at N=4096 -- fewer than the 132 SMs -- so it starves the machine there even
// though its higher A/B reuse wins at N=15360. This shape-dependent BM choice is
// exactly the kind of thing the design-notes selector wants.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_bf16.h>

using bf16 = __nv_bfloat16;

// ---- tile config (shared by wgmma kernel and mma.sync baseline) ----
#ifndef WGMMA_MS
#define WGMMA_MS 1
#endif
#ifndef WGMMA_NS
#define WGMMA_NS 3
#endif
static constexpr int MS = WGMMA_MS;  // m64 wgmma slabs per warpgroup (BM = 64*MS)
static constexpr int BM = 64 * MS;
static constexpr int BN = 128;   // n128
static constexpr int BK = 64;    // staged K per pipeline stage -> 4 k16 substeps
static constexpr int WG = 128;   // one warpgroup / 4 warps
static constexpr int NS = WGMMA_NS;  // cp.async pipeline depth
static constexpr int NACC = 64 * BN / WG;  // 64 f32 accumulators / thread / slab
static constexpr int KSUB = BK / 16;       // 4 wgmma k16 substeps per stage
static constexpr int KCHUNK = BK / 8;      // 8 16-byte chunks per row (= 128 B)
static constexpr int LBO = 16;             // logical k-core byte stride (128B swz)
static constexpr int SBO = 1024;           // 8 rows x 128 B swizzle atom
static constexpr int SWZ_128B = 1;         // descriptor bits[63:62]
static constexpr int ATILE = BM * BK;      // bf16 per A stage
static constexpr int BTILE = BN * BK;      // bf16 per B stage
static constexpr int WSMEM = NS * (ATILE + BTILE) * (int)sizeof(bf16) + 1024;  // +align slack

// baseline (mma.sync) uses plain padded row-major smem instead
static constexpr int PAD = 8;
static constexpr int STRD = BK + PAD;
static constexpr int BSMEM = NS * (BM + BN) * STRD * (int)sizeof(bf16);

// ================= wgmma primitives =================

// 64-bit smem matrix descriptor. bits[13:0] start>>4, [29:16] LBO>>4,
// [45:32] SBO>>4, [51:49] matrix base offset, [63:62] swizzle mode (0 = none).
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

// 128B-swizzled element offset of the 16-byte chunk c (= k/8) of logical row `row`
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

// one m64n128k16 .f32.bf16.bf16, both operands from smem (SS form).
// scaleD: 0 -> D = A*B (seeds the accumulator, no zeroing pass needed)
//         1 -> D = A*B + D.  scale-a = scale-b = +1, trans-a = trans-b = 0 (K-major).
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

// stage a [rows x BK] tile into the 128B-swizzled smem layout.
// out-of-range rows / k zero-fill via the cp.async src-size operand.
// thread order (c fastest) keeps each 8-thread 16B phase inside one 128 B row -> no bank conflicts.
__device__ __forceinline__ void stage_swz(bf16* dst, const bf16* __restrict__ src, int tid,
                                          int rows, int row0, int kbase, int R, int K) {
    const int chunks = rows * KCHUNK;
    for (int L = tid; L < chunks; L += WG) {
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

// ================= WGMMA kernel =================
// NS-deep cp.async K pipeline; wgmma group ks-1 is retired (wg_wait<1>) before the
// buffer it read is refilled, so the wgmma of step ks overlaps the loads for ks+NS-1.
__global__ __launch_bounds__(WG) void k_wgmma(float* __restrict__ C, const bf16* __restrict__ A,
                                              const bf16* __restrict__ B, int M, int N, int K) {
    extern __shared__ bf16 smem_raw[];
    // 128B swizzle needs the tile base 1024 B aligned so the HW address-bit swizzle
    // ([6:4] chunk ^ [9:7] row) matches swz_off() and matrix-base-offset can stay 0.
    bf16* smem = smem_raw + ((1024 - ((unsigned)__cvta_generic_to_shared(smem_raw) & 1023)) & 1023) / 2;
    bf16* As = smem;                    // [NS][BM][BK] 128B-swizzled
    bf16* Bs = smem + NS * ATILE;       // [NS][BN][BK] 128B-swizzled
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;

    const int mtiles = (M + BM - 1) / BM;
    const int ntiles = (N + BN - 1) / BN;
    const int total = mtiles * ntiles;
    const int ksteps = (K + BK - 1) / BK;

    for (int t = blockIdx.x; t < total; t += gridDim.x) {
        const int tm = (t / ntiles) * BM, tn = (t % ntiles) * BN;

        float acc[MS][NACC];  // one m64n128 accumulator set per slab

        // prologue: NS-1 stages in flight
#pragma unroll
        for (int s = 0; s < NS - 1; s++) {
            if (s < ksteps) {
                stage_swz(As + s * ATILE, A, tid, BM, tm, s * BK, M, K);
                stage_swz(Bs + s * BTILE, B, tid, BN, tn, s * BK, N, K);
            }
            cp_commit();
        }

        for (int ks = 0; ks < ksteps; ks++) {
            const int cur = ks % NS;
            cp_wait<NS - 2>();  // stage `ks` landed
            __syncthreads();

            bf16* Ac = As + cur * ATILE;
            bf16* Bc = Bs + cur * BTILE;
            wg_fence();
#pragma unroll
            for (int sub = 0; sub < KSUB; sub++) {
                // scaleD=0 on the very first issue seeds the accumulators (no zeroing pass)
                const int sd = (ks == 0 && sub == 0) ? 0 : 1;
                const uint64_t db = make_desc(Bc + sub * 16, LBO, SBO);
#pragma unroll
                for (int s = 0; s < MS; s++)
                    wgmma_m64n128k16(acc[s], make_desc(Ac + s * 64 * BK + sub * 16, LBO, SBO), db,
                                     sd);
            }
            wg_commit();
            wg_wait<1>();  // group ks-1 done -> buffer (ks-1)%NS free to refill

            const int nxt = ks + NS - 1;
            if (nxt < ksteps) {
                const int nb = nxt % NS;
                stage_swz(As + nb * ATILE, A, tid, BM, tm, nxt * BK, M, K);
                stage_swz(Bs + nb * BTILE, B, tid, BN, tn, nxt * BK, N, K);
            }
            cp_commit();
        }
        wg_wait<0>();

        // epilogue. warpgroup m64nN f32 accumulator -> (row,col):
        //   warp w owns rows [16w..16w+15]; for n-block j (0..BN/8-1):
        //     acc[4j+0] -> (16w + lane/4,     8j + (lane%4)*2 + 0)
        //     acc[4j+1] -> (16w + lane/4,     8j + (lane%4)*2 + 1)
        //     acc[4j+2] -> (16w + lane/4 + 8, 8j + (lane%4)*2 + 0)
        //     acc[4j+3] -> (16w + lane/4 + 8, 8j + (lane%4)*2 + 1)
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
        __syncthreads();  // tile done; next grid-stride tile reuses smem
    }
}

// ================= mma.sync m16n8k16 baseline =================
// Same BM/BN/BK tiling, same NS-deep cp.async pipeline, plain padded row-major
// smem + ldmatrix, so the delta measured is the tensor-core issue path only.
// 4 warps in a 2x2 grid over the 64x128 tile -> each warp 32x64 = 2x8 fragments
// x 4 f32 = 64 accumulators/thread, identical register budget to the wgmma path.
static constexpr int WM = BM / 2, WN = BN / 2, MFRAG = WM / 16, NFRAG = WN / 8;

__device__ __forceinline__ void ldm_x4(unsigned (&r)[4], const void* s) {
    unsigned a = (unsigned)__cvta_generic_to_shared(s);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldm_x2(unsigned (&r)[2], const void* s) {
    unsigned a = (unsigned)__cvta_generic_to_shared(s);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}
__device__ __forceinline__ void mma16816(float (&d)[4], const unsigned (&a)[4],
                                         const unsigned (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// plain padded row-major stage (baseline only)
__device__ __forceinline__ void stage_rm(bf16* dst, const bf16* __restrict__ src, int tid,
                                         int rows, int row0, int kbase, int R, int K) {
    const int chunks = rows * (BK / 8);
    for (int L = tid; L < chunks; L += WG) {
        const int row = L / (BK / 8), kc = L - row * (BK / 8);
        const int gr = row0 + row, gk = kbase + kc * 8;
        int bytes = 0;
        const bf16* g = src;
        if (gr < R && gk < K) {
            g = src + (size_t)gr * K + gk;
            const int rem = K - gk;
            bytes = rem >= 8 ? 16 : rem * 2;
        }
        cp16(&dst[row * STRD + kc * 8], g, bytes);
    }
}

__global__ __launch_bounds__(WG) void k_mma(float* __restrict__ C, const bf16* __restrict__ A,
                                            const bf16* __restrict__ B, int M, int N, int K) {
    extern __shared__ bf16 smem[];
    bf16* As = smem;
    bf16* Bs = smem + NS * BM * STRD;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp >> 1, wn = warp & 1;

    const int mtiles = (M + BM - 1) / BM, ntiles = (N + BN - 1) / BN;
    const int total = mtiles * ntiles, ksteps = (K + BK - 1) / BK;

    for (int t = blockIdx.x; t < total; t += gridDim.x) {
        const int tm = (t / ntiles) * BM, tn = (t % ntiles) * BN;
        float acc[MFRAG][NFRAG][4];
#pragma unroll
        for (int i = 0; i < MFRAG; i++)
#pragma unroll
            for (int j = 0; j < NFRAG; j++)
#pragma unroll
                for (int q = 0; q < 4; q++) acc[i][j][q] = 0.f;

#pragma unroll
        for (int s = 0; s < NS - 1; s++) {
            if (s < ksteps) {
                stage_rm(As + s * BM * STRD, A, tid, BM, tm, s * BK, M, K);
                stage_rm(Bs + s * BN * STRD, B, tid, BN, tn, s * BK, N, K);
            }
            cp_commit();
        }

        for (int ks = 0; ks < ksteps; ks++) {
            const int cur = ks % NS;
            cp_wait<NS - 2>();
            __syncthreads();
            bf16* Ac = As + cur * BM * STRD;
            bf16* Bc = Bs + cur * BN * STRD;
#pragma unroll
            for (int kf = 0; kf < BK; kf += 16) {
                unsigned af[MFRAG][4], bf_[NFRAG][2];
#pragma unroll
                for (int i = 0; i < MFRAG; i++)
                    ldm_x4(af[i], &Ac[(wm * WM + i * 16 + (lane % 16)) * STRD + kf +
                                      (lane / 16) * 8]);
#pragma unroll
                for (int j = 0; j < NFRAG; j++)
                    ldm_x2(bf_[j], &Bc[(wn * WN + j * 8 + (lane & 7)) * STRD + kf +
                                       ((lane >> 3) & 1) * 8]);
#pragma unroll
                for (int i = 0; i < MFRAG; i++)
#pragma unroll
                    for (int j = 0; j < NFRAG; j++) mma16816(acc[i][j], af[i], bf_[j]);
            }
            __syncthreads();
            const int nxt = ks + NS - 1;
            if (nxt < ksteps) {
                const int nb = nxt % NS;
                stage_rm(As + nb * BM * STRD, A, tid, BM, tm, nxt * BK, M, K);
                stage_rm(Bs + nb * BN * STRD, B, tid, BN, tn, nxt * BK, N, K);
            }
            cp_commit();
        }

#pragma unroll
        for (int i = 0; i < MFRAG; i++) {
            const int r0 = tm + wm * WM + i * 16 + (lane >> 2), r1 = r0 + 8;
#pragma unroll
            for (int j = 0; j < NFRAG; j++) {
                const int c = tn + wn * WN + j * 8 + (lane & 3) * 2;
                if (r0 < M) {
                    if (c < N) C[(size_t)r0 * N + c] = acc[i][j][0];
                    if (c + 1 < N) C[(size_t)r0 * N + c + 1] = acc[i][j][1];
                }
                if (r1 < M) {
                    if (c < N) C[(size_t)r1 * N + c] = acc[i][j][2];
                    if (c + 1 < N) C[(size_t)r1 * N + c + 1] = acc[i][j][3];
                }
            }
        }
        __syncthreads();
    }
}

// ================= host: oracle, validation, benchmark =================
static uint32_t g_xs = 0x1234567u;
static float frand() {  // xorshift32 -> [-1,1)
    g_xs ^= g_xs << 13; g_xs ^= g_xs >> 17; g_xs ^= g_xs << 5;
    return ((g_xs >> 8) * (1.0f / 8388608.0f)) - 1.0f;
}
static float of_bf16(bf16 b) { return __bfloat162float(b); }

#define CK(x)                                                                                   \
    do {                                                                                        \
        cudaError_t e_ = (x);                                                                   \
        if (e_ != cudaSuccess) {                                                                \
            printf("CUDA ERR %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));           \
            exit(1);                                                                            \
        }                                                                                       \
    } while (0)

// f32 oracle over the SAME bf16 inputs the kernel reads
static void oracle(std::vector<float>& C, const std::vector<bf16>& A, const std::vector<bf16>& B,
                   int M, int N, int K) {
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            const bf16* a = &A[(size_t)m * K];
            const bf16* b = &B[(size_t)n * K];
            float s = 0.f;
            for (int k = 0; k < K; k++) s += of_bf16(a[k]) * of_bf16(b[k]);
            C[(size_t)m * N + n] = s;
        }
}

static int grid_for(const void* fn, int smem, int total) {
    int occ = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, WG, smem);
    if (occ < 1) occ = 1;
    int cap = 132 * occ;
    return total < cap ? total : cap;
}

struct Res { double relL2; double tf_wgmma; double tf_mma; };

static Res run_shape(int M, int N, int K, bool bench, int* fail) {
    Res r{0, 0, 0};
    std::vector<bf16> hA((size_t)M * K), hB((size_t)N * K);
    for (auto& x : hA) x = __float2bfloat16(frand());
    for (auto& x : hB) x = __float2bfloat16(frand());

    bf16 *dA, *dB; float* dC;
    CK(cudaMalloc(&dA, hA.size() * sizeof(bf16)));
    CK(cudaMalloc(&dB, hB.size() * sizeof(bf16)));
    CK(cudaMalloc(&dC, (size_t)M * N * sizeof(float)));
    CK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(bf16), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(bf16), cudaMemcpyHostToDevice));

    CK(cudaFuncSetAttribute(k_wgmma, cudaFuncAttributeMaxDynamicSharedMemorySize, WSMEM));
    CK(cudaFuncSetAttribute(k_mma, cudaFuncAttributeMaxDynamicSharedMemorySize, BSMEM));
    const int total = ((M + BM - 1) / BM) * ((N + BN - 1) / BN);
    const int gw = grid_for((const void*)k_wgmma, WSMEM, total);
    const int gm = grid_for((const void*)k_mma, BSMEM, total);

    CK(cudaMemset(dC, 0, (size_t)M * N * sizeof(float)));
    k_wgmma<<<gw, WG, WSMEM>>>(dC, dA, dB, M, N, K);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    std::vector<float> hC((size_t)M * N), ref((size_t)M * N);
    CK(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float), cudaMemcpyDeviceToHost));
    oracle(ref, hA, hB, M, N, K);
    double num = 0, den = 0;
    for (size_t i = 0; i < hC.size(); i++) {
        double d = (double)hC[i] - (double)ref[i];
        num += d * d; den += (double)ref[i] * (double)ref[i];
    }
    r.relL2 = sqrt(num / (den + 1e-30));
    const bool pass = r.relL2 < 3e-3;
    if (!pass) *fail = 1;

    // baseline correctness too (so the speedup compares two verified kernels)
    CK(cudaMemset(dC, 0, (size_t)M * N * sizeof(float)));
    k_mma<<<gm, WG, BSMEM>>>(dC, dA, dB, M, N, K);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float), cudaMemcpyDeviceToHost));
    double n2 = 0;
    for (size_t i = 0; i < hC.size(); i++) {
        double d = (double)hC[i] - (double)ref[i];
        n2 += d * d;
    }
    const double bl = sqrt(n2 / (den + 1e-30));
    if (!(bl < 3e-3)) *fail = 1;

    printf("  (%4d,%5d,%5d)  wgmma relL2=%.3e %s   | mma.sync relL2=%.3e %s\n", M, N, K, r.relL2,
           pass ? "PASS" : "FAIL", bl, bl < 3e-3 ? "PASS" : "FAIL");

    if (bench) {
        cudaEvent_t a, b;
        CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        const int iters = 100, warm = 10;
        const double flop = 2.0 * M * N * K;
        float ms;
        for (int i = 0; i < warm; i++) k_wgmma<<<gw, WG, WSMEM>>>(dC, dA, dB, M, N, K);
        CK(cudaDeviceSynchronize()); CK(cudaEventRecord(a));
        for (int i = 0; i < iters; i++) k_wgmma<<<gw, WG, WSMEM>>>(dC, dA, dB, M, N, K);
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        CK(cudaEventElapsedTime(&ms, a, b));
        r.tf_wgmma = flop / (ms / 1e3 / iters) / 1e12;
        const double wms = ms / iters;

        for (int i = 0; i < warm; i++) k_mma<<<gm, WG, BSMEM>>>(dC, dA, dB, M, N, K);
        CK(cudaDeviceSynchronize()); CK(cudaEventRecord(a));
        for (int i = 0; i < iters; i++) k_mma<<<gm, WG, BSMEM>>>(dC, dA, dB, M, N, K);
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        CK(cudaEventElapsedTime(&ms, a, b));
        r.tf_mma = flop / (ms / 1e3 / iters) / 1e12;
        printf("    wgmma %7.3f ms  %6.1f TFLOP/s | mma.sync %7.3f ms  %6.1f TFLOP/s | speedup "
               "%.2fx\n",
               wms, r.tf_wgmma, ms / iters, r.tf_mma, r.tf_wgmma / r.tf_mma);
        cudaEventDestroy(a); cudaEventDestroy(b);
    }
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return r;
}

int main() {
    printf("== WGMMA m64n128k16 bf16 TN-GEMM probe (H100 sm_90a) ==\n");
    printf("   BM=%d BN=%d BK=%d stages=%d  smem wgmma=%.1f KiB / mma=%.1f KiB\n", BM, BN, BK, NS,
           WSMEM / 1024.0, BSMEM / 1024.0);
    printf("CORRECTNESS (relL2 < 3e-3, f32 oracle over the same bf16 inputs):\n");
    int fail = 0;
    run_shape(64, 128, 16, false, &fail);
    run_shape(128, 256, 64, false, &fail);
    run_shape(512, 4096, 3840, false, &fail);
    run_shape(512, 15360, 3840, false, &fail);
    run_shape(200, 4096, 3840, false, &fail);
    printf("RESULT: %s\n\n", fail ? "FAIL" : "PASS");

    if (!fail) {
        printf("BENCHMARK (Gemma-4 prefill, 100 iters + 10 warmup):\n");
        run_shape(512, 4096, 3840, true, &fail);
        run_shape(512, 15360, 3840, true, &fail);
    }
    return fail;
}
