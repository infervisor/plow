// DeepSeek-V4.1 sm_90a GEMMs.
//
// gemm_w8a8: C[M,N] = sum_kb (A8[M,kb] . W8[N,kb]^T) * sa[m,kb] * sw[n/32,kb]  -- kernel.py fp8_gemm at
//   block 32. One m16n8k32 e4m3 MMA step is exactly one 32-wide K block, so each step's partial is
//   rescaled into a separate fp32 accumulator, as the reference does.
// gemm_bf16w: C[M,N] = A_bf16[M,K] . W[N,K]^T with W bf16, or e4m3 dequantized by a [32,32] ue8m0 grid
//   (exact in bf16), fp32 accumulate. Batched over `batch` with element strides.
// gemm_f32: SIMT fp32, A bf16|f32, W bf16|f32, C f32.
#include "dsv41_common.cuh"

#define CP_ASYNC_16(dst, src, bytes)                                                                  \
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(bytes))
#define CP_ASYNC_COMMIT() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT(n) asm volatile("cp.async.wait_group %0;\n" ::"n"(n))

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return (uint32_t)__cvta_generic_to_shared(p);
}

__device__ __forceinline__ void mma_e4m3(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void mma_bf16(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// ---------------------------------------------------------------------------------------------------
// W8A8. Tile 64(M) x 128(N), 4 warps as 2x2, warp tile 32x64. A three-stage cp.async ring stages
// 128 K per step (four m16n8k32 steps, each one 32-wide scale block, promoted into the fp32
// accumulator with sa[m][kb] * sw[n/32][kb] as kernel.py's fp8_gemm does). Rows padded to 144 B so
// the fragment loads land on distinct banks. Split-K over whole stages. K % 128 == 0.
// Dynamic shared memory: W8_SMEM.
#define W8_BM 64
#define W8_BN 128
#define W8_ST 3
#define W8_KS 128
#define W8_LD 144
#define W8_SMEM (W8_ST * (W8_BM + W8_BN) * W8_LD)
DSV_EXTERN void __launch_bounds__(128)
    dsv_gemm_w8a8(void* __restrict__ C, const uint8_t* __restrict__ A, const uint8_t* __restrict__ sa,
                  const uint8_t* __restrict__ W, const uint8_t* __restrict__ sw, int M, int N, int K,
                  long long ldc, int c_f32, int ksplit, float* __restrict__ part) {
    extern __shared__ __align__(16) uint8_t w8_smem[];
    uint8_t* As = w8_smem;                          // [ST][64][144]
    uint8_t* Ws = w8_smem + W8_ST * W8_BM * W8_LD;  // [ST][128][144]
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp >> 1, wn = warp & 1;
    const int g = lane >> 2, t4 = lane & 3;
    const int m0 = blockIdx.y * W8_BM, n0 = blockIdx.x * W8_BN;
    const int KB = K >> 5, KS = K / W8_KS;
    const int split = blockIdx.z;
    const int ks0 = (int)((long long)KS * split / ksplit), ks1 = (int)((long long)KS * (split + 1) / ksplit);

    auto load = [&](int slot, int ks) {
#pragma unroll
        for (int i = 0; i < 4; i++) {  // A: 64 rows x 8 chunks of 16 B
            const int c = tid + i * 128, r = c >> 3, off = (c & 7) * 16;
            const int gm = m0 + r;
            CP_ASYNC_16(smem_u32(&As[(slot * W8_BM + r) * W8_LD + off]), A + (long long)min(gm, M - 1) * K + ks * W8_KS + off, gm < M ? 16 : 0);
        }
#pragma unroll
        for (int i = 0; i < 8; i++) {  // W: 128 rows x 8 chunks of 16 B
            const int c = tid + i * 128, r = c >> 3, off = (c & 7) * 16;
            const int gn = n0 + r;
            CP_ASYNC_16(smem_u32(&Ws[(slot * W8_BN + r) * W8_LD + off]), W + (long long)min(gn, N - 1) * K + ks * W8_KS + off, gn < N ? 16 : 0);
        }
        CP_ASYNC_COMMIT();
    };

    float acc[2][8][4];
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 8; j++)
#pragma unroll
            for (int k = 0; k < 4; k++) acc[i][j][k] = 0.f;
    int rows[2][2];
#pragma unroll
    for (int i = 0; i < 2; i++) {
        rows[i][0] = min(m0 + wm * 32 + i * 16 + g, M - 1);
        rows[i][1] = min(m0 + wm * 32 + i * 16 + g + 8, M - 1);
    }
    const int NB = (N + 31) >> 5;
    const int nb0 = min((n0 + wn * 64) >> 5, NB - 1), nb1 = min((n0 + wn * 64 + 32) >> 5, NB - 1);

    if (ks0 < ks1) load(0, ks0);
    if (ks0 + 1 < ks1) load(1, ks0 + 1); else CP_ASYNC_COMMIT();
    for (int ks = ks0; ks < ks1; ks++) {
        CP_ASYNC_WAIT(1);
        __syncthreads();
        if (ks + 2 < ks1) load((ks - ks0 + 2) % W8_ST, ks + 2); else CP_ASYNC_COMMIT();
        const int slot = (ks - ks0) % W8_ST;
#pragma unroll
        for (int kk = 0; kk < W8_KS / 32; kk++) {
            const int kb = ks * (W8_KS / 32) + kk;
            uint32_t af[2][4], bfr[8][2];
#pragma unroll
            for (int i = 0; i < 2; i++) {
                const uint8_t* base = &As[(slot * W8_BM + wm * 32 + i * 16 + g) * W8_LD + kk * 32 + t4 * 4];
                af[i][0] = *(const uint32_t*)(base);
                af[i][1] = *(const uint32_t*)(base + 8 * W8_LD);
                af[i][2] = *(const uint32_t*)(base + 16);
                af[i][3] = *(const uint32_t*)(base + 8 * W8_LD + 16);
            }
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const uint8_t* base = &Ws[(slot * W8_BN + wn * 64 + j * 8 + g) * W8_LD + kk * 32 + t4 * 4];
                bfr[j][0] = *(const uint32_t*)(base);
                bfr[j][1] = *(const uint32_t*)(base + 16);
            }
            const float s_w0 = e8m0_to_f(sw[(long long)nb0 * KB + kb]);
            const float s_w1 = e8m0_to_f(sw[(long long)nb1 * KB + kb]);
#pragma unroll
            for (int i = 0; i < 2; i++) {
                const float sa0 = e8m0_to_f(sa[(long long)rows[i][0] * KB + kb]);
                const float sa1 = e8m0_to_f(sa[(long long)rows[i][1] * KB + kb]);
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    float t[4] = {0.f, 0.f, 0.f, 0.f};
                    mma_e4m3(t, af[i], bfr[j]);
                    const float swj = j < 4 ? s_w0 : s_w1;
                    acc[i][j][0] += t[0] * sa0 * swj;
                    acc[i][j][1] += t[1] * sa0 * swj;
                    acc[i][j][2] += t[2] * sa1 * swj;
                    acc[i][j][3] += t[3] * sa1 * swj;
                }
            }
        }
    }
    CP_ASYNC_WAIT(0);

#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int m = m0 + wm * 32 + i * 16 + g + h * 8;
            if (m >= M) continue;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const int n = n0 + wn * 64 + j * 8 + t4 * 2;
                if (n >= N) continue;
                const float v0 = acc[i][j][h * 2], v1 = acc[i][j][h * 2 + 1];
                if (ksplit > 1) {
                    float* pp = part + ((long long)split * M + m) * N + n;
                    pp[0] = v0;
                    if (n + 1 < N) pp[1] = v1;
                } else if (c_f32) {
                    float* cp = (float*)C + (long long)m * ldc + n;
                    cp[0] = v0;
                    if (n + 1 < N) cp[1] = v1;
                } else {
                    bf16* cp = (bf16*)C + (long long)m * ldc + n;
                    cp[0] = f2bf(v0);
                    if (n + 1 < N) cp[1] = f2bf(v1);
                }
            }
        }
}

// ---------------------------------------------------------------------------------------------------
// bf16 activations x (bf16 | e4m3[32,32]-scaled) weights. Tile 64 x 128 x 32 (two k16 MMA steps), 4 warps
// 2x2, warp tile 32x64. The weight tile is dequantized to bf16 while being staged into shared memory.
// Rows padded to 40 bf16 (80 B) for conflict-free 32-bit fragment loads.
#define BW_LD 40
DSV_EXTERN void __launch_bounds__(128)
    dsv_gemm_bf16w(void* __restrict__ C, const bf16* __restrict__ A, const void* __restrict__ W,
                   const uint8_t* __restrict__ sw, int M, int N, int K, long long lda, long long ldc,
                   int w_fp8, int c_f32, long long a_bstride, long long w_bstride, long long sw_bstride,
                   long long c_bstride, int ksplit, float* __restrict__ part) {
    __shared__ __align__(16) bf16 As[64 * BW_LD];
    __shared__ __align__(16) bf16 Ws[128 * BW_LD];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp >> 1, wn = warp & 1;
    const int g = lane >> 2, t4 = lane & 3;
    const int m0 = blockIdx.y * 64, n0 = blockIdx.x * 128;
    // grid.z = batch * ksplit: batch entry bz, K split `split` owning K blocks [kb0, kb1)
    const int bz = blockIdx.z / ksplit, split = blockIdx.z % ksplit;
    const int nbatch = gridDim.z / ksplit;
    A += bz * a_bstride;
    const int KB = K >> 5;
    const int kb0 = (int)((long long)KB * split / ksplit), kb1 = (int)((long long)KB * (split + 1) / ksplit);
    const int NB = (N + 31) >> 5;

    float acc[2][8][4];
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 8; j++)
#pragma unroll
            for (int k = 0; k < 4; k++) acc[i][j][k] = 0.f;

    for (int kb = kb0; kb < kb1; kb++) {
        // A: 64 rows x 32 bf16; each thread 16 elements (2 x uint4)
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int ch = tid + i * 128;  // 256 chunks of 8 bf16
            const int r = ch >> 2, c = (ch & 3) * 8;
            const int gm = m0 + r;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (gm < M) v = *(const uint4*)(A + (long long)gm * lda + kb * 32 + c);
            *(uint4*)(&As[r * BW_LD + c]) = v;
        }
        // W: 128 rows x 32
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int ch = tid + i * 128;  // 512 chunks of 8 elements
            const int r = ch >> 2, c = (ch & 3) * 8;
            const int gn = n0 + r;
            uint4 out = make_uint4(0, 0, 0, 0);
            if (gn < N) {
                if (w_fp8) {
                    const uint8_t* wp = (const uint8_t*)W + bz * w_bstride + (long long)gn * K + kb * 32 + c;
                    const uint2 raw = *(const uint2*)wp;
                    const float s = e8m0_to_f(sw[bz * sw_bstride + (long long)(gn >> 5) * KB + kb]);
                    const uint8_t* b = (const uint8_t*)&raw;
                    bf16 o[8];
#pragma unroll
                    for (int e = 0; e < 8; e++) o[e] = f2bf(e4m3_to_f(b[e]) * s);
                    out = *(const uint4*)o;
                } else {
                    out = *(const uint4*)((const bf16*)W + bz * w_bstride + (long long)gn * K + kb * 32 + c);
                }
            }
            *(uint4*)(&Ws[r * BW_LD + c]) = out;
        }
        __syncthreads();
#pragma unroll
        for (int ks = 0; ks < 2; ks++) {
            uint32_t af[2][4], bfr[8][2];
#pragma unroll
            for (int i = 0; i < 2; i++) {
                const bf16* base = &As[(wm * 32 + i * 16 + g) * BW_LD + ks * 16 + t4 * 2];
                af[i][0] = *(const uint32_t*)(base);
                af[i][1] = *(const uint32_t*)(base + 8 * BW_LD);
                af[i][2] = *(const uint32_t*)(base + 8);
                af[i][3] = *(const uint32_t*)(base + 8 * BW_LD + 8);
            }
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const bf16* base = &Ws[(wn * 64 + j * 8 + g) * BW_LD + ks * 16 + t4 * 2];
                bfr[j][0] = *(const uint32_t*)(base);
                bfr[j][1] = *(const uint32_t*)(base + 8);
            }
#pragma unroll
            for (int i = 0; i < 2; i++)
#pragma unroll
                for (int j = 0; j < 8; j++) mma_bf16(acc[i][j], af[i], bfr[j]);
        }
        __syncthreads();
    }
    (void)NB;
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int m = m0 + wm * 32 + i * 16 + g + h * 8;
            if (m >= M) continue;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const int n = n0 + wn * 64 + j * 8 + t4 * 2;
                if (n >= N) continue;
                const float v0 = acc[i][j][h * 2], v1 = acc[i][j][h * 2 + 1];
                if (ksplit > 1) {
                    float* pp = part + (((long long)split * nbatch + bz) * M + m) * N + n;
                    pp[0] = v0;
                    if (n + 1 < N) pp[1] = v1;
                } else if (c_f32) {
                    float* cp = (float*)C + bz * c_bstride + (long long)m * ldc + n;
                    cp[0] = v0;
                    if (n + 1 < N) cp[1] = v1;
                } else {
                    bf16* cp = (bf16*)C + bz * c_bstride + (long long)m * ldc + n;
                    cp[0] = f2bf(v0);
                    if (n + 1 < N) cp[1] = f2bf(v1);
                }
            }
        }
}

// Split-K reduction: C[b][m][n] = sum over splits in order of part[s][b][m][n] (deterministic).
// part is [ksplit][batch][M][N] f32; C row stride ldc, batch stride c_bstride, bf16 or f32.
DSV_EXTERN void dsv_splitk_reduce(void* __restrict__ C, const float* __restrict__ part, int ksplit, int batch, int M, int N,
                                  long long ldc, long long c_bstride, int c_f32) {
    const long long total = (long long)batch * M * N;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
        float v = 0.f;
        for (int s = 0; s < ksplit; s++) v += part[(long long)s * total + i];
        const int b = (int)(i / ((long long)M * N));
        const long long r = i % ((long long)M * N);
        const int m = (int)(r / N), n = (int)(r % N);
        const long long o = b * c_bstride + (long long)m * ldc + n;
        if (c_f32) ((float*)C)[o] = v;
        else ((bf16*)C)[o] = f2bf(v);
    }
}

// ---------------------------------------------------------------------------------------------------
// fp32 SIMT GEMM: C[M,N] (f32) = A[M,K] . W[N,K]^T. Tile 64x64x16, 256 threads, 4x4 per thread.
template <bool A_BF16, bool W_BF16>
__device__ __forceinline__ void gemm_f32_body(float* __restrict__ C, const void* __restrict__ A,
                                              const void* __restrict__ W, int M, int N, int K,
                                              long long lda, long long ldc) {
    __shared__ float As[16][64 + 4];
    __shared__ float Ws[16][64 + 4];
    const int tid = threadIdx.x;
    const int tx = tid & 15, ty = tid >> 4;
    const int m0 = blockIdx.y * 64, n0 = blockIdx.x * 64;
    float acc[4][4] = {};
    for (int k0 = 0; k0 < K; k0 += 16) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int e = tid + i * 256;  // 1024 = 64 rows x 16 k
            const int r = e >> 4, c = e & 15;
            const int gm = m0 + r, gk = k0 + c;
            float av = 0.f;
            if (gm < M && gk < K)
                av = A_BF16 ? bf2f(((const bf16*)A)[(long long)gm * lda + gk])
                            : ((const float*)A)[(long long)gm * lda + gk];
            As[c][r] = av;
            const int gn = n0 + r;
            float wv = 0.f;
            if (gn < N && gk < K)
                wv = W_BF16 ? bf2f(((const bf16*)W)[(long long)gn * K + gk]) : ((const float*)W)[(long long)gn * K + gk];
            Ws[c][r] = wv;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < 16; kk++) {
            float a[4], w[4];
#pragma unroll
            for (int i = 0; i < 4; i++) {
                a[i] = As[kk][ty * 4 + i];
                w[i] = Ws[kk][tx * 4 + i];
            }
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) acc[i][j] = fmaf(a[i], w[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int m = m0 + ty * 4 + i;
        if (m >= M) continue;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int n = n0 + tx * 4 + j;
            if (n < N) C[(long long)m * ldc + n] = acc[i][j];
        }
    }
}
// fp32 dot form for small output grids (M*N small, K large -- the mHC mixes are 24 x 20480):
// one block per output element, the whole block reducing over K. grid = (N, M).
DSV_EXTERN void __launch_bounds__(256) dsv_gemm_f32_dot(float* __restrict__ C, const void* __restrict__ A,
                                                        const void* __restrict__ W, int M, int N, int K, long long lda,
                                                        long long ldc, int a_bf16, int w_bf16) {
    __shared__ float red[32];
    const int n = blockIdx.x, m = blockIdx.y;
    float acc = 0.f;
    if (a_bf16 && w_bf16) {
        const bf16* a = (const bf16*)A + (long long)m * lda;
        const bf16* w = (const bf16*)W + (long long)n * K;
        for (int k = threadIdx.x; k < K; k += blockDim.x) acc = fmaf(bf2f(a[k]), bf2f(w[k]), acc);
    } else if (a_bf16) {
        const bf16* a = (const bf16*)A + (long long)m * lda;
        const float* w = (const float*)W + (long long)n * K;
        for (int k = threadIdx.x; k < K; k += blockDim.x) acc = fmaf(bf2f(a[k]), w[k], acc);
    } else if (w_bf16) {
        const float* a = (const float*)A + (long long)m * lda;
        const bf16* w = (const bf16*)W + (long long)n * K;
        for (int k = threadIdx.x; k < K; k += blockDim.x) acc = fmaf(a[k], bf2f(w[k]), acc);
    } else {
        const float* a = (const float*)A + (long long)m * lda;
        const float* w = (const float*)W + (long long)n * K;
        for (int k = threadIdx.x; k < K; k += blockDim.x) acc = fmaf(a[k], w[k], acc);
    }
    acc = block_sum(acc, red);
    if (threadIdx.x == 0) C[(long long)m * ldc + n] = acc;
}

// fp32 row form for few outputs over many rows (the mHC mixes in prefill: N = 24, K = 20480):
// each block owns 4 rows and all N <= 32 outputs, accumulating in registers, so every A row is read
// once and W (N x K) streams from L2. grid = ceil(M / 4).
#define F32R_ROWS 4
DSV_EXTERN void __launch_bounds__(256) dsv_gemm_f32_rows(float* __restrict__ C, const void* __restrict__ A,
                                                         const void* __restrict__ W, int M, int N, int K, long long lda,
                                                         long long ldc, int a_bf16, int w_bf16) {
    __shared__ float red[8][F32R_ROWS][32];
    const int m0 = blockIdx.x * F32R_ROWS;
    float acc[F32R_ROWS][32];
#pragma unroll
    for (int r = 0; r < F32R_ROWS; r++)
#pragma unroll
        for (int n = 0; n < 32; n++) acc[r][n] = 0.f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        float a[F32R_ROWS];
#pragma unroll
        for (int r = 0; r < F32R_ROWS; r++) {
            const int m = min(m0 + r, M - 1);
            a[r] = a_bf16 ? bf2f(((const bf16*)A)[(long long)m * lda + k]) : ((const float*)A)[(long long)m * lda + k];
        }
#pragma unroll
        for (int n = 0; n < 32; n++) {
            if (n >= N) break;
            const float w = w_bf16 ? bf2f(((const bf16*)W)[(long long)n * K + k]) : ((const float*)W)[(long long)n * K + k];
#pragma unroll
            for (int r = 0; r < F32R_ROWS; r++) acc[r][n] = fmaf(a[r], w, acc[r][n]);
        }
    }
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
#pragma unroll
    for (int r = 0; r < F32R_ROWS; r++)
#pragma unroll
        for (int n = 0; n < 32; n++) {
            const float v = warp_sum(acc[r][n]);
            if (lane == 0) red[wid][r][n] = v;
        }
    __syncthreads();
    for (int i = threadIdx.x; i < F32R_ROWS * N; i += blockDim.x) {
        const int r = i / N, n = i % N;
        const int m = m0 + r;
        if (m >= M) continue;
        float s = 0.f;
        for (int w = 0; w < (int)(blockDim.x >> 5); w++) s += red[w][r][n];
        C[(long long)m * ldc + n] = s;
    }
}

DSV_EXTERN void __launch_bounds__(256) dsv_gemm_f32(float* C, const void* A, const void* W, int M, int N, int K,
                                                    long long lda, long long ldc, int a_bf16, int w_bf16) {
    if (a_bf16) {
        if (w_bf16) gemm_f32_body<true, true>(C, A, W, M, N, K, lda, ldc);
        else gemm_f32_body<true, false>(C, A, W, M, N, K, lda, ldc);
    } else {
        if (w_bf16) gemm_f32_body<false, true>(C, A, W, M, N, K, lda, ldc);
        else gemm_f32_body<false, false>(C, A, W, M, N, K, lda, ldc);
    }
}
