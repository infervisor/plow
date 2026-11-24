// Hopper (sm_90a) fp8 warpgroup-MMA GEMM with per-32 MX promotion:
//   C[M][N] = sum_kb  sa[m][kb] * sw[n/32][kb] * (A8[m][kb*32..] . W8[n][kb*32..])
// A8 / W8 are e4m3, sa / sw ue8m0 (kernel.py fp8_gemm with act_quant(x, 32, "ue8m0") inputs). Each
// 32-wide K block is one wgmma m64n128k32 into a scratch accumulator (scale-d = 0) that the consumer
// folds into the fp32 result with its block scale -- the per-block promotion the reference does.
//
// CTA: 384 threads = warpgroup 0 (producer: cp.async of A, W and both scale strips straight into the
// 128B-swizzled stage) + warpgroups 1, 2 (consumers, 64 rows each). Tile 128 x 128, 128 K per stage,
// F8_ST stages, PERSISTENT: the CTA walks tiles first, first + step, ... with one stage counter, so
// the producer streams the next tile while the consumers run the last one's epilogue. Consumers wait
// the stage's cp.async mbarrier themselves, fence the async proxy, run the four k32 blocks and
// release the slot. K % 128 == 0.
//
// Measured limit (H200, q_a at 8k rows): the fold, not the wgmma wait, bounds the tile -- ~513
// TFLOP/s with it, ~1080 without it (per-block wait kept). Overlapping a warpgroup's fold with its
// own next wgmma needs a second scratch set and was slower in every form tried (n64 halves, n96 or
// n64 tiles with two scratch sets, a fold mutex between the warpgroups): 1.3-2x slower.
#pragma once
#include "op_wg_sm90.cuh"

#define F8_BM 128
#define F8_BN 128
#define F8_BK 128
#define F8_ST 5
#define F8_TILE_BYTES (F8_BM * F8_BK)  // A (and B) stage, e4m3, 16 KiB
#define F8_SMEM (F8_ST * (2 * F8_TILE_BYTES + F8_BM * 4 + 64) + 2 * F8_ST * 8 + 1024)

namespace plow_wg8 {
using namespace plow_wg;

__device__ __forceinline__ void cp_async4(uint32_t dst, const void* src, int bytes) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(dst), "l"(src), "r"(bytes) : "memory");
}

__device__ __forceinline__ void wg_mma_e4m3_m64n128k32(float* d, uint64_t da, uint64_t db) {
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, "
        "%64, %65, 0, 1, 1;\n"
        "}\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7]), "=f"(d[8]), "=f"(d[9]), "=f"(d[10]), "=f"(d[11]), "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15]), "=f"(d[16]), "=f"(d[17]), "=f"(d[18]), "=f"(d[19]), "=f"(d[20]), "=f"(d[21]), "=f"(d[22]), "=f"(d[23]), "=f"(d[24]), "=f"(d[25]), "=f"(d[26]), "=f"(d[27]), "=f"(d[28]), "=f"(d[29]), "=f"(d[30]), "=f"(d[31]), "=f"(d[32]), "=f"(d[33]), "=f"(d[34]), "=f"(d[35]), "=f"(d[36]), "=f"(d[37]), "=f"(d[38]), "=f"(d[39]), "=f"(d[40]), "=f"(d[41]), "=f"(d[42]), "=f"(d[43]), "=f"(d[44]), "=f"(d[45]), "=f"(d[46]), "=f"(d[47]), "=f"(d[48]), "=f"(d[49]), "=f"(d[50]), "=f"(d[51]), "=f"(d[52]), "=f"(d[53]), "=f"(d[54]), "=f"(d[55]), "=f"(d[56]), "=f"(d[57]), "=f"(d[58]), "=f"(d[59]), "=f"(d[60]), "=f"(d[61]), "=f"(d[62]), "=f"(d[63])
        : "l"(da), "l"(db));
}

// 2^(e - 127) for a ue8m0 byte
__device__ __forceinline__ float e8m0(uint32_t e) { return __uint_as_float(e << 23); }

// The op: groups of block-diagonal GEMMs. Group g: C[:, g*N ..] (ldc) = A8[:, g*K ..] (lda bytes) x
// W8[g*N .., K]^T, sa [rows][las bytes] at column g*K/32, sw [(g*N + n)/32][K/32]. Rows 0 .. M.
struct Fp8Job {
    bf16* C;
    long long ldc;
    const uint8_t* A8;
    long long lda;
    const uint8_t* sa;
    long long las;
    const uint8_t* W8;
    const uint8_t* sw;
    int M, N, K, groups;
};

// Tiles t0, t0 + step, ... of groups x ceil(M/128) x ceil(N/128) (N fastest). Every thread of the CTA
// calls it; it ends on a block barrier.
__device__ __forceinline__ void gemm_fp8(const Fp8Job& job, unsigned t0, unsigned step) {
    extern __shared__ __align__(1024) uint8_t wg_smem_raw[];
    uint8_t* smem = (uint8_t*)(((uintptr_t)wg_smem_raw + 1023) & ~(uintptr_t)1023);
    uint8_t* As = smem;                          // [ST][128 x 128 B]
    uint8_t* Bs = As + F8_ST * F8_TILE_BYTES;    // [ST][128 x 128 B]
    uint8_t* SA = Bs + F8_ST * F8_TILE_BYTES;    // [ST][128 rows][4 kb]
    uint8_t* SW = SA + F8_ST * F8_BM * 4;        // [ST][4 row blocks][4 kb] (+ pad to 64)
    uint64_t* bar_ld = (uint64_t*)(SW + F8_ST * 64);
    uint64_t* bar_empty = bar_ld + F8_ST;

    const int N = job.N, K = job.K, KS = K / F8_BK, KB = K >> 5;
    const unsigned tn = (N + F8_BN - 1) / F8_BN, tm = (job.M + F8_BM - 1) / F8_BM, n_tiles = tn * tm * job.groups;
    const int tid = threadIdx.x, wg = tid >> 7, lt = tid & 127;
    if (tid == 0) {
        for (int s = 0; s < F8_ST; s++) {
            mbar_init(&bar_ld[s], 128);
            mbar_init(&bar_empty[s], 256);
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();

    if (wg == 0) {
        const int c = lt & 7;
        unsigned it = 0;
        for (unsigned tile = t0; tile < n_tiles; tile += step) {
            const unsigned gi = tile / (tn * tm), r = tile % (tn * tm);
            const int row0 = (int)(r / tn) * F8_BM, n0 = (int)(r % tn) * F8_BN;
            const uint8_t* arow[8];
            const uint8_t* wrow[8];
            int av[8], wv[8];
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const int rr = (lt >> 3) + i * 16;
                av[i] = row0 + rr < job.M ? 16 : 0;
                arow[i] = job.A8 + (long long)(av[i] ? row0 + rr : 0) * job.lda + (long long)gi * K + c * 16;
                wv[i] = n0 + rr < N ? 16 : 0;
                wrow[i] = job.W8 + ((long long)gi * N + (wv[i] ? n0 + rr : 0)) * K + c * 16;
            }
            const int sav = row0 + lt < job.M ? 4 : 0;
            const uint8_t* sarow = job.sa + (long long)(sav ? row0 + lt : 0) * job.las + (long long)gi * KB;
            const int swv = lt < 4 && n0 + lt * 32 < N ? 4 : 0;
            const uint8_t* swrow = job.sw + ((long long)gi * (N >> 5) + (swv ? (n0 >> 5) + lt : 0)) * KB;
            for (int ks = 0; ks < KS; ks++, it++) {
                const int slot = it % F8_ST;
                if (it >= F8_ST) mbar_wait(&bar_empty[slot], ((it / F8_ST) - 1) & 1);
                const uint32_t a_base = smem_u32(As + slot * F8_TILE_BYTES), b_base = smem_u32(Bs + slot * F8_TILE_BYTES);
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    const int rr = (lt >> 3) + i * 16;
                    wg_cp_async16(a_base + sw128(rr, c), arow[i] + ks * F8_BK, av[i]);
                    wg_cp_async16(b_base + sw128(rr, c), wrow[i] + ks * F8_BK, wv[i]);
                }
                cp_async4(smem_u32(SA + slot * F8_BM * 4 + lt * 4), sarow + ks * 4, sav);
                if (lt < 4) cp_async4(smem_u32(SW + slot * 64 + lt * 4), swrow + ks * 4, swv);
                mbar_cp_async_arrive(&bar_ld[slot]);
            }
        }
    } else {
        const int cw = wg - 1, w = lt >> 5, lane = lt & 31, g = lane >> 2, t4 = lane & 3;
        const int r0 = cw * 64 + w * 16 + g;  // this thread's tile rows r0, r0 + 8
        unsigned it = 0;
        for (unsigned tile = t0; tile < n_tiles; tile += step) {
            const unsigned gi = tile / (tn * tm), r = tile % (tn * tm);
            const int row0 = (int)(r / tn) * F8_BM, n0 = (int)(r % tn) * F8_BN;
            float d[64], t[64];
#pragma unroll
            for (int i = 0; i < 64; i++) d[i] = 0.f;
            for (int ks = 0; ks < KS; ks++, it++) {
                const int slot = it % F8_ST;
                mbar_wait(&bar_ld[slot], (it / F8_ST) & 1);
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                const uint32_t s0 = *(const uint32_t*)(SA + slot * F8_BM * 4 + r0 * 4);
                const uint32_t s1 = *(const uint32_t*)(SA + slot * F8_BM * 4 + (r0 + 8) * 4);
                const uint4 swq = *(const uint4*)(SW + slot * 64);
                const uint32_t sws[4] = {swq.x, swq.y, swq.z, swq.w};
                const uint64_t da = wg_desc(smem_u32(As + slot * F8_TILE_BYTES + cw * 64 * 128));
                const uint64_t db = wg_desc(smem_u32(Bs + slot * F8_TILE_BYTES));
#pragma unroll
                for (int kk = 0; kk < 4; kk++) {
                    wg_fence();
                    wg_mma_e4m3_m64n128k32(t, da + 2 * kk, db + 2 * kk);
                    wg_commit();
                    wg_wait<0>();
                    if (kk == 3) mbar_arrive(&bar_empty[slot]);
                    const float fa0 = e8m0((s0 >> (8 * kk)) & 0xffu), fa1 = e8m0((s1 >> (8 * kk)) & 0xffu);
#pragma unroll
                    for (int cb = 0; cb < 4; cb++) {
                        const float fw = e8m0((sws[cb] >> (8 * kk)) & 0xffu);
                        const float f0 = fa0 * fw, f1 = fa1 * fw;
#pragma unroll
                        for (int j = cb * 4; j < cb * 4 + 4; j++) {
                            d[j * 4 + 0] = fmaf(t[j * 4 + 0], f0, d[j * 4 + 0]);
                            d[j * 4 + 1] = fmaf(t[j * 4 + 1], f0, d[j * 4 + 1]);
                            d[j * 4 + 2] = fmaf(t[j * 4 + 2], f1, d[j * 4 + 2]);
                            d[j * 4 + 3] = fmaf(t[j * 4 + 3], f1, d[j * 4 + 3]);
                        }
                    }
                }
            }
            bf16* C = job.C + (long long)gi * N;
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int rr = row0 + r0 + h * 8;
                if (rr >= job.M) continue;
#pragma unroll
                for (int j = 0; j < 16; j++) {
                    const int col = n0 + j * 8 + t4 * 2;
                    if (col < N) *(__nv_bfloat162*)(C + (long long)rr * job.ldc + col) = __floats2bfloat162_rn(d[j * 4 + h * 2], d[j * 4 + h * 2 + 1]);
                }
            }
        }
    }
    __syncthreads();
}

}  // namespace plow_wg8
