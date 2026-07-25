// wgmma_fp8_probe.cu — standalone Hopper (sm_90a) FP8 e4m3 WGMMA GEMM microkernel.
//
// GOAL: prove a real warpgroup wgmma.mma_async.m64nNk32.f32.e4m3.e4m3 mainloop is
// numerically correct against an independent f32 CPU oracle that dequantizes the SAME
// e4m3 bytes, then benchmark TFLOP/s vs a warp-level mma.sync.m16n8k32.e4m3 baseline.
//
// Contract (plow w8a8 prefill linear, op_gemm.cuh pgm_mma_fp8_k32):
//   C[m,n] = scale_a*scale_b * sum_k A[m,k]*B[n,k]
//   A e4m3 [M,K] k-contiguous, B e4m3 [N,K] k-contiguous, C f32 [M,N]  (TN GEMM = A.B^T)
//
// Build:
//   env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin nvcc -std=c++17 -arch=sm_90a -O3 \
//     -I runtime/common -I runtime/nvidia -include cstdint \
//     runtime/nvidia/experiments/wgmma_fp8_probe.cu -o <bin>
// Run (GPU serialized):  flock /tmp/plow_gpu.lock <bin>
//
// WGMMA smem layout (derived from CUTLASS cute make_gmma_desc, Major::K, INTERLEAVE):
//   canonical K-major no-swizzle atom = 8 rows x 16B core matrices, row-major inside.
//   packing: u128_index(mn,k) = ((mn/8)*KB + k/16)*8 + (mn%8)  (mn-block major, K minor), KB=BK/16
//   => stride_byte_offset (SBO) = KB*8 u128, leading_byte_offset (LBO) = 8 u128, layout=0.
//   k32 subgroup ks of a staged tile starts 2 core matrices (256 B) in => desc base + ks*256.
//   Both A[64,32] and B[N,32] tiles use the identical K-major descriptor (fp8 is K-major only,
//   no transpose args). cp.async 16B lines land directly on core-matrix rows; zero-fill masks tails.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <thread>
#include <cuda_runtime.h>
#include <cuda_fp8.h>

#define CHK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA ERR %s: %s\n",#x,cudaGetErrorString(e)); exit(1);} }while(0)

// ------- WGMMA tile geometry (one warpgroup = 128 threads per CTA) -------
#define BM 64          // M tile = wgmma M (fixed 64)
#define BN 128         // N tile = wgmma N (= wgmma instruction N)
#define BK 64          // smem K depth per stage; BK/32 wgmma k32 issues per stage
#define NSTAGE 4       // cp.async pipeline depth (buffers); >= PREFETCH+2 for 1 wgmma in flight
#define WG_THREADS 128
#define KB (BK / 16)   // core matrices along K per mn-block
#define KSUB (BK / 32) // k32 wgmma sub-steps per staged tile
#define NACC (BN / 2)  // f32 accumulator registers per thread
#define PREFETCH (NSTAGE - 2)   // cp.async prefetch distance (stages)
#if BN == 64
#define WGMMA_TILE(a, da, db, sd) wgmma_m64n64k32(a, da, db, sd)
#elif BN == 128
#define WGMMA_TILE(a, da, db, sd) wgmma_m64n128k32(a, da, db, sd)
#else
#error "no wgmma helper for this BN"
#endif

// Hopper FP8 WGMMA accumulates with reduced internal precision (see report). Promoting the
// wgmma accumulator into a plain f32 (CUDA-core) shadow every PROMOTE_STAGES stages recovers
// accuracy — this is DeepGEMM's two-level accumulation. 0 = disabled.
#ifndef PROMOTE_STAGES
#define PROMOTE_STAGES 0
#endif

// smem byte offset for a K-major INTERLEAVE (no-swizzle) fp8 tile.
// core matrix = 8 mn-rows x 16 K-bytes, row-major; blocks ordered mn-major / K-minor.
__device__ __forceinline__ uint32_t kmaj_off(int mn, int k) {
    int nn = mn >> 3, r = mn & 7;
    int kb = k >> 4, c = k & 15;
    return ((uint32_t)((nn * KB + kb) * 8 + r) << 4) + (uint32_t)c;   // *16 bytes + byte-in-line
}

// Build the 64-bit GMMA matrix descriptor (Major::K, INTERLEAVE) for a smem tile.
// start_address = smem_addr>>4 (16B units), LBO/SBO already in 16B (uint128) units, layout=0.
__device__ __forceinline__ uint64_t make_desc_k(const void* smem_ptr) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    const uint64_t LBO = 8;        // leading byte offset (stride to next K core-matrix), u128 units
    const uint64_t SBO = KB * 8;   // stride byte offset  (stride to next 8-row mn block), u128 units
    uint64_t d = 0;
    d |= (uint64_t)((addr >> 4) & 0x3FFFu);        // [0,14)  start address
    d |= (LBO & 0x3FFFu) << 16;                    // [16,30) leading byte offset
    d |= (SBO & 0x3FFFu) << 32;                    // [32,46) stride byte offset
    // base_offset [49,52) = 0 ; layout_type [62,64) = 0 (INTERLEAVE)
    return d;
}

__device__ __forceinline__ void cpasync16(uint32_t dst, const void* src, int valid_bytes) {
    // cp.async.cg 16B with source-size (zero-fill) — bytes past valid_bytes become 0.
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::
                 "r"(dst), "l"(src), "r"(valid_bytes));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void cp_wait0()  { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

// wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3  (SS: both operands from smem descriptors).
// scale_a=scale_b=+1 (immediate); scale_d runtime predicate (1 => accumulate onto d).
__device__ __forceinline__ void wgmma_m64n64k32(float* d, uint64_t desc_a, uint64_t desc_b, int scale_d) {
    asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %34, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
      "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
      " %8,  %9,  %10, %11, %12, %13, %14, %15, "
      " %16, %17, %18, %19, %20, %21, %22, %23, "
      " %24, %25, %26, %27, %28, %29, %30, %31}, "
      " %32, %33, p, 1, 1;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
        "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]),"+f"(d[11]),
        "+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),
        "+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),
        "+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),
        "+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),
        "+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31])
      : "l"(desc_a), "l"(desc_b), "r"(scale_d));
}
__device__ __forceinline__ void wgmma_m64n128k32(float* d, uint64_t desc_a, uint64_t desc_b, int scale_d) {
    asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %66, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 "
      "{%0, %1, %2, %3, %4, %5, %6, %7,"
      " %8, %9, %10, %11, %12, %13, %14, %15,"
      " %16, %17, %18, %19, %20, %21, %22, %23,"
      " %24, %25, %26, %27, %28, %29, %30, %31,"
      " %32, %33, %34, %35, %36, %37, %38, %39,"
      " %40, %41, %42, %43, %44, %45, %46, %47,"
      " %48, %49, %50, %51, %52, %53, %54, %55,"
      " %56, %57, %58, %59, %60, %61, %62, %63"
      "}, %64, %65, p, 1, 1;\n"
      "}\n"
      :
        "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
        "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
        "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
        "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
        "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
      : "l"(desc_a), "l"(desc_b), "r"(scale_d));
}

__device__ __forceinline__ void wgmma_fence()  { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_wait0()  { asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void wgmma_wait() {
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory");
}

// ---------------- WGMMA GEMM kernel ----------------
__global__ void __launch_bounds__(WG_THREADS)
wgmma_gemm(const uint8_t* __restrict__ A, const uint8_t* __restrict__ B, float* __restrict__ C,
           int M, int N, int K, float scale_a, float scale_b) {
    extern __shared__ __align__(128) uint8_t smem[];
    const int ATILE = BM * BK, BTILE = BN * BK;
    uint8_t* As = smem;                        // [NSTAGE][BM][BK] K-major INTERLEAVE
    uint8_t* Bs = smem + NSTAGE * ATILE;       // [NSTAGE][BN][BK] K-major INTERLEAVE
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int tilesM = (M + BM - 1) / BM, tilesN = (N + BN - 1) / BN;
    const int nt = tilesM * tilesN;
    const int nk = (K + BK - 1) / BK;

    for (int tile = blockIdx.x; tile < nt; tile += gridDim.x) {
        const int tm = (tile / tilesN) * BM, tn = (tile % tilesN) * BN;
        float acc[NACC];
#pragma unroll
        for (int i = 0; i < NACC; i++) acc[i] = 0.f;
#if PROMOTE_STAGES
        float pacc[NACC];
#pragma unroll
        for (int i = 0; i < NACC; i++) pacc[i] = 0.f;
        int reset_acc = 1, scnt = 0;
#endif
        // stage `ki` of this tile into buffer `buf`
        auto stage = [&](int ki, int buf) {
            const int k0 = ki * BK;
            uint32_t Aa = (uint32_t)__cvta_generic_to_shared(As + buf * ATILE);
            uint32_t Ba = (uint32_t)__cvta_generic_to_shared(Bs + buf * BTILE);
            for (int L = tid; L < BM * KB; L += WG_THREADS) {
                int mn = L / KB, kb = L % KB;
                int row = tm + mn, kk = k0 + kb * 16;
                int vb = (row < M) ? (K - kk) : 0;
                vb = vb < 0 ? 0 : (vb > 16 ? 16 : vb);
                const void* g = vb ? (const void*)(A + (size_t)row * K + kk) : (const void*)A;
                cpasync16(Aa + kmaj_off(mn, kb * 16), g, vb);
            }
            for (int L = tid; L < BN * KB; L += WG_THREADS) {
                int mn = L / KB, kb = L % KB;
                int col = tn + mn, kk = k0 + kb * 16;
                int vb = (col < N) ? (K - kk) : 0;
                vb = vb < 0 ? 0 : (vb > 16 ? 16 : vb);
                const void* g = vb ? (const void*)(B + (size_t)col * K + kk) : (const void*)B;
                cpasync16(Ba + kmaj_off(mn, kb * 16), g, vb);
            }
        };

        cp_wait0(); __syncthreads();          // drain previous tile's pipeline
        for (int i = 0; i < PREFETCH; i++) { if (i < nk) stage(i, i % NSTAGE); cp_commit(); }

        for (int i = 0; i < nk; i++) {
            cp_wait<PREFETCH - 1>();          // buffer i%NSTAGE has landed
            __syncthreads();
            int pf = i + PREFETCH;            // buffer (i+PREFETCH)%NSTAGE was last read by group i-2
            if (pf < nk) stage(pf, pf % NSTAGE);
            cp_commit();

            const uint8_t* Ab = As + (i % NSTAGE) * ATILE;
            const uint8_t* Bb = Bs + (i % NSTAGE) * BTILE;
            wgmma_fence();
#pragma unroll
            for (int ks = 0; ks < KSUB; ks++) {
                // k32 subgroup ks starts 2 core matrices (2*128 B) into the staged tile
                uint64_t da = make_desc_k(Ab + ks * 256);
                uint64_t db = make_desc_k(Bb + ks * 256);
#if PROMOTE_STAGES
                int sd = (reset_acc && ks == 0) ? 0 : 1;
                WGMMA_TILE(acc, da, db, sd);
#else
                WGMMA_TILE(acc, da, db, 1);
#endif
            }
            wgmma_commit();
#if PROMOTE_STAGES
            wgmma_wait<0>();                  // promotion must read acc this iteration
            reset_acc = 0;
            if (++scnt % PROMOTE_STAGES == 0) {   // fold into the f32 shadow, restart wgmma acc
#pragma unroll
                for (int j = 0; j < NACC; j++) pacc[j] += acc[j];
                reset_acc = 1;
            }
#else
            wgmma_wait<1>();                  // keep one wgmma group in flight (async overlap)
#endif
        }
        wgmma_wait<0>();                      // drain before the epilogue reads acc
#if PROMOTE_STAGES
        if (!reset_acc) {
#pragma unroll
            for (int j = 0; j < NACC; j++) pacc[j] += acc[j];
        }
#pragma unroll
        for (int j = 0; j < NACC; j++) acc[j] = pacc[j];
#endif

        // epilogue: wgmma f32 C-fragment -> global, per CuTe CLayout_64xN
        //   row = 16*warp + lane/4 + 8*hi ; col = 8*g + 2*(lane%4) + lo ; reg = 4g + 2hi + lo
        const float s = scale_a * scale_b;
#pragma unroll
        for (int g = 0; g < BN / 8; g++)
#pragma unroll
            for (int hi = 0; hi < 2; hi++)
#pragma unroll
                for (int lo = 0; lo < 2; lo++) {
                    int r = 4 * g + 2 * hi + lo;
                    int row = warp * 16 + (lane >> 2) + 8 * hi;
                    int col = 8 * g + 2 * (lane & 3) + lo;
                    int m = tm + row, n = tn + col;
                    if (m < M && n < N) C[(size_t)m * N + n] = acc[r] * s;
                }
    }
}

// ---------------- mma.sync m16n8k32 e4m3 baseline (warp-level) ----------------
// Same BM=64,BN=64 tile, 4 warps over M (16 rows each), each warp loops 8 n8 groups.
// Fragment layout: empirically-verified (fp8_verify.cu) — A lane L rows (L>>2),(L>>2)+8, k=8*(L&3)..+7.
__device__ __forceinline__ void mma_m16n8k32(float d[4], const uint32_t a[4], const uint32_t b[2]) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__global__ void __launch_bounds__(WG_THREADS)
mma_gemm(const uint8_t* __restrict__ A, const uint8_t* __restrict__ B, float* __restrict__ C,
         int M, int N, int K, float scale_a, float scale_b) {
    extern __shared__ __align__(128) uint8_t smem[];
    uint8_t* As = smem;                 // [BM][BK] row-major (stride 32)
    uint8_t* Bs = smem + BM * BK;       // [BN][BK] row-major (stride 32)
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int tilesM = (M + BM - 1) / BM, tilesN = (N + BN - 1) / BN;
    const int nt = tilesM * tilesN;

    for (int tile = blockIdx.x; tile < nt; tile += gridDim.x) {
        const int tm = (tile / tilesN) * BM, tn = (tile % tilesN) * BN;
        float acc[BN / 8][4];
#pragma unroll
        for (int j = 0; j < BN / 8; j++) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;

        for (int k0 = 0; k0 < K; k0 += BK) {
            for (int L = tid; L < BM * KB; L += WG_THREADS) {
                int mn = L / KB, kb = L % KB;
                int row = tm + mn, kk = k0 + kb * 16;
                int vb = (row < M) ? (K - kk) : 0;
                vb = vb < 0 ? 0 : (vb > 16 ? 16 : vb);
                const void* g = vb ? (const void*)(A + (size_t)row * K + kk) : (const void*)A;
                cpasync16((uint32_t)__cvta_generic_to_shared(As + mn * BK + kb * 16), g, vb);
            }
            for (int L = tid; L < BN * KB; L += WG_THREADS) {
                int mn = L / KB, kb = L % KB;
                int col = tn + mn, kk = k0 + kb * 16;
                int vb = (col < N) ? (K - kk) : 0;
                vb = vb < 0 ? 0 : (vb > 16 ? 16 : vb);
                const void* g = vb ? (const void*)(B + (size_t)col * K + kk) : (const void*)B;
                cpasync16((uint32_t)__cvta_generic_to_shared(Bs + mn * BK + kb * 16), g, vb);
            }
            cp_commit();
            cp_wait0();
            __syncthreads();

            int rlo = warp * 16 + (lane >> 2), rhi = rlo + 8;
#pragma unroll
            for (int ks = 0; ks < KSUB; ks++) {
                int kb = ks * 32 + 8 * (lane & 3);
                uint32_t a[4];
                a[0] = *(const uint32_t*)(As + rlo * BK + kb);
                a[2] = *(const uint32_t*)(As + rlo * BK + kb + 4);
                a[1] = *(const uint32_t*)(As + rhi * BK + kb);
                a[3] = *(const uint32_t*)(As + rhi * BK + kb + 4);
#pragma unroll
                for (int j = 0; j < BN / 8; j++) {
                    int col = j * 8 + (lane >> 2);
                    uint32_t b[2];
                    b[0] = *(const uint32_t*)(Bs + col * BK + kb);
                    b[1] = *(const uint32_t*)(Bs + col * BK + kb + 4);
                    mma_m16n8k32(acc[j], a, b);
                }
            }
            __syncthreads();
        }

        const float s = scale_a * scale_b;
#pragma unroll
        for (int j = 0; j < BN / 8; j++)
#pragma unroll
            for (int r = 0; r < 4; r++) {
                int row = warp * 16 + (lane >> 2) + 8 * (r >> 1);
                int col = j * 8 + 2 * (lane & 3) + (r & 1);
                int m = tm + row, n = tn + col;
                if (m < M && n < N) C[(size_t)m * N + n] = acc[j][r] * s;
            }
    }
}

// ---------------- host helpers ----------------
static inline uint8_t f2e4m3(float f) { __nv_fp8_e4m3 v(f); return (uint8_t)v.__x; }
static inline float   e4m32f(uint8_t b){ __nv_fp8_e4m3 v; v.__x = (__nv_fp8_storage_t)b; return (float)v; }

// f32 CPU oracle: dequantize the SAME e4m3 bytes, C = scale * sum_k A[m,k]*B[n,k]. Threaded over M.
static void cpu_oracle(const uint8_t* A, const uint8_t* B, std::vector<float>& C,
                       int M, int N, int K, float scale_a, float scale_b) {
    std::vector<float> Af((size_t)M * K), Bf((size_t)N * K);
    for (size_t i = 0; i < (size_t)M * K; i++) Af[i] = e4m32f(A[i]);
    for (size_t i = 0; i < (size_t)N * K; i++) Bf[i] = e4m32f(B[i]);
    const float s = scale_a * scale_b;
    unsigned T = std::thread::hardware_concurrency(); if (!T) T = 8; if ((int)T > M) T = M;
    std::vector<std::thread> ths;
    auto work = [&](int m0, int m1) {
        for (int m = m0; m < m1; m++) {
            const float* a = &Af[(size_t)m * K];
            for (int n = 0; n < N; n++) {
                const float* b = &Bf[(size_t)n * K];
                float acc = 0.f;
                for (int k = 0; k < K; k++) acc += a[k] * b[k];
                C[(size_t)m * N + n] = acc * s;
            }
        }
    };
    int per = (M + (int)T - 1) / (int)T;
    for (unsigned t = 0; t < T; t++) {
        int m0 = (int)t * per, m1 = (m0 + per > M) ? M : m0 + per;
        if (m0 < m1) ths.emplace_back(work, m0, m1);
    }
    for (auto& th : ths) th.join();
}

static double relL2(const std::vector<float>& dev, const std::vector<float>& ref) {
    double num = 0, den = 0;
    for (size_t i = 0; i < ref.size(); i++) {
        double d = (double)dev[i] - (double)ref[i];
        num += d * d; den += (double)ref[i] * (double)ref[i];
    }
    return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
}

struct Shape { int M, N, K; };
static int launch_blocks(int nt) { int b = 132 * 8; return b < nt ? b : nt; }

int main() {
    const float scale_a = 0.5f, scale_b = 2.0f;
    Shape shapes[] = {{64,128,32},{128,256,64},{512,4096,3840},{512,15360,3840},{200,4096,3840}};
    const int NS = (int)(sizeof(shapes)/sizeof(shapes[0]));
    printf("=== WGMMA FP8 e4m3 probe (sm_90a) — scale_a=%.2f scale_b=%.2f ===\n", scale_a, scale_b);
    printf("wgmma m64n%dk32 | tile %dx%dx%d | %d-stage cp.async (prefetch %d) | promote=%d\n",
           BN, BM, BN, BK, NSTAGE, PREFETCH, PROMOTE_STAGES);
    printf("smem K-major INTERLEAVE (no swizzle): LBO=8 SBO=%d (u128 units)\n\n", KB * 8);
    printf("%-22s %-12s %-12s %s\n", "shape (M,N,K)", "wgmma relL2", "mma relL2", "gate<6e-3");
    bool all_pass = true;
    CHK(cudaFuncSetAttribute(wgmma_gemm, cudaFuncAttributeMaxDynamicSharedMemorySize, 200000));
    CHK(cudaFuncSetAttribute(mma_gemm,   cudaFuncAttributeMaxDynamicSharedMemorySize, 200000));
    const size_t smem_w = (size_t)NSTAGE * (BM * BK + BN * BK);  // wgmma: NSTAGE-buffered
    const size_t smem_m = (size_t)(BM * BK + BN * BK);           // mma.sync baseline: single-buffered

    for (int si = 0; si < NS; si++) {
        int M = shapes[si].M, N = shapes[si].N, K = shapes[si].K;
        size_t szA = (size_t)M * K, szB = (size_t)N * K, szC = (size_t)M * N;
        std::vector<uint8_t> hA(szA), hB(szB);
        srand(1234 + si);
        for (size_t i = 0; i < szA; i++) hA[i] = f2e4m3((rand() / (float)RAND_MAX) * 2.f - 1.f);
        for (size_t i = 0; i < szB; i++) hB[i] = f2e4m3((rand() / (float)RAND_MAX) * 2.f - 1.f);

        uint8_t *dA, *dB; float* dC;
        CHK(cudaMalloc(&dA, szA)); CHK(cudaMalloc(&dB, szB)); CHK(cudaMalloc(&dC, szC * 4));
        CHK(cudaMemcpy(dA, hA.data(), szA, cudaMemcpyHostToDevice));
        CHK(cudaMemcpy(dB, hB.data(), szB, cudaMemcpyHostToDevice));
        int nt = ((M+BM-1)/BM) * ((N+BN-1)/BN);

        std::vector<float> hC(szC), hCm(szC), ref(szC);
        CHK(cudaMemset(dC, 0, szC * 4));
        wgmma_gemm<<<launch_blocks(nt), WG_THREADS, smem_w>>>(dA, dB, dC, M, N, K, scale_a, scale_b);
        CHK(cudaGetLastError()); CHK(cudaDeviceSynchronize());
        CHK(cudaMemcpy(hC.data(), dC, szC * 4, cudaMemcpyDeviceToHost));

        CHK(cudaMemset(dC, 0, szC * 4));
        mma_gemm<<<launch_blocks(nt), WG_THREADS, smem_m>>>(dA, dB, dC, M, N, K, scale_a, scale_b);
        CHK(cudaGetLastError()); CHK(cudaDeviceSynchronize());
        CHK(cudaMemcpy(hCm.data(), dC, szC * 4, cudaMemcpyDeviceToHost));

        cpu_oracle(hA.data(), hB.data(), ref, M, N, K, scale_a, scale_b);
        double e = relL2(hC, ref), em = relL2(hCm, ref);
        bool pass = e < 6e-3;
        all_pass &= pass;
        char buf[64]; snprintf(buf, sizeof buf, "(%d,%d,%d)", M, N, K);
        printf("%-22s %-12.4e %-12.4e %s\n", buf, e, em, pass ? "PASS" : "FAIL");

        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }
    printf("\nRESULT: %s\n", all_pass ? "PASS" : "FAIL");

    // ---------------- benchmark ----------------
    printf("\n=== benchmark: TFLOP/s (100 iter / 10 warmup) ===\n");
    printf("%-20s %-14s %-14s %s\n", "shape", "wgmma TF/s", "mma.sync TF/s", "speedup");
    Shape bench[] = {{512,4096,3840},{512,15360,3840}};
    for (int bi = 0; bi < 2; bi++) {
        int M = bench[bi].M, N = bench[bi].N, K = bench[bi].K;
        size_t szA=(size_t)M*K, szB=(size_t)N*K, szC=(size_t)M*N;
        std::vector<uint8_t> hA(szA), hB(szB);
        for (size_t i=0;i<szA;i++) hA[i]=f2e4m3(0.1f);
        for (size_t i=0;i<szB;i++) hB[i]=f2e4m3(0.1f);
        uint8_t *dA,*dB; float* dC;
        CHK(cudaMalloc(&dA,szA)); CHK(cudaMalloc(&dB,szB)); CHK(cudaMalloc(&dC,szC*4));
        CHK(cudaMemcpy(dA,hA.data(),szA,cudaMemcpyHostToDevice));
        CHK(cudaMemcpy(dB,hB.data(),szB,cudaMemcpyHostToDevice));
        int nt = ((M+BM-1)/BM) * ((N+BN-1)/BN), blk = launch_blocks(nt);
        double flops = 2.0 * M * N * K;
        cudaEvent_t ev0, ev1; cudaEventCreate(&ev0); cudaEventCreate(&ev1);
        auto timed = [&](int which)->double {
            for (int i=0;i<10;i++) {
                if (which==0) wgmma_gemm<<<blk,WG_THREADS,smem_w>>>(dA,dB,dC,M,N,K,scale_a,scale_b);
                else          mma_gemm  <<<blk,WG_THREADS,smem_m>>>(dA,dB,dC,M,N,K,scale_a,scale_b);
            }
            CHK(cudaDeviceSynchronize());
            cudaEventRecord(ev0);
            for (int i=0;i<100;i++) {
                if (which==0) wgmma_gemm<<<blk,WG_THREADS,smem_w>>>(dA,dB,dC,M,N,K,scale_a,scale_b);
                else          mma_gemm  <<<blk,WG_THREADS,smem_m>>>(dA,dB,dC,M,N,K,scale_a,scale_b);
            }
            cudaEventRecord(ev1); CHK(cudaEventSynchronize(ev1));
            float ms=0; cudaEventElapsedTime(&ms,ev0,ev1);
            return flops / ((double)ms/100.0*1e-3) / 1e12;   // TFLOP/s
        };
        double tw = timed(0), tm = timed(1);
        char buf[64]; snprintf(buf,sizeof buf,"(%d,%d,%d)",M,N,K);
        printf("%-20s %-14.1f %-14.1f %.2fx\n", buf, tw, tm, tw/tm);
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }
    return all_pass ? 0 : 1;
}
