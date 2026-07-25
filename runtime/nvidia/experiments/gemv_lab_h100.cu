// gemv_lab_h100.cu — E2 of the 26B/H100 beat-vLLM campaign.
//
// E1 (hbm_ceiling_h100.cu) established on this box:
//   achievable read BW 3707 GB/s; at 1 block/SM with 8 loads in flight per thread
//   a pure read still reaches 2490 GB/s (62%). vLLM's 26B decode achieves 1579 GB/s.
//   plow's decode GEMV achieves ~816 GB/s. So the gap is NOT occupancy — beating
//   vLLM needs only 1579 GB/s, and 2490 is reachable at the megakernel's own occupancy.
//
// This lab finds which GEMV formulation gets there, on the REAL 26B decode shapes,
// weights cold (L2 flushed), M=1, at 1/2/3 blocks per SM.
//
// Variants (all compute the same C[n] = dot(x, W[n,:])):
//   A_noun   probe-style: one warp/row, no unroll, x from global   (prior probe's arm)
//   B_un8    production gemv_rows: 8 W+x loads pre-issued, x from global
//   C_smemx  x staged in smem once per block, 8 W loads pre-issued
//   D_rowR   C + warp owns R rows at once (R independent W streams, x reused)
//   E_hfma2  D with packed bf16x2 FMA instead of per-element float converts
//
//   nvcc -arch=sm_90a -O3 -Xptxas -v -o gl gemv_lab_h100.cu
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define CHK(x)                                                                             \
    do {                                                                                   \
        cudaError_t e = (x);                                                               \
        if (e != cudaSuccess) {                                                            \
            printf("CUDA ERR %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e));          \
            exit(1);                                                                       \
        }                                                                                  \
    } while (0)

static const double HBM_ACHIEVABLE = 3707.0;  // measured E1
#define BLOCK 256u
#define WPB (BLOCK / 32u)
#define GV_STEP 256u  // 32 lanes * 8 elems
#define MAXK 4096u    // largest K in the decode set (o_proj)

typedef struct {
    uint4 v;
} bf16v8;
__device__ __forceinline__ bf16v8 ld8(const __nv_bfloat16* p) {
    bf16v8 r;
    r.v = *(const uint4*)p;
    return r;
}
__device__ __forceinline__ bf16v8 zero8() {
    bf16v8 r;
    r.v = make_uint4(0, 0, 0, 0);
    return r;
}
__device__ __forceinline__ float dot8(const bf16v8& a, const bf16v8& b, float acc) {
    const __nv_bfloat16* A = (const __nv_bfloat16*)&a.v;
    const __nv_bfloat16* B = (const __nv_bfloat16*)&b.v;
#pragma unroll
    for (int j = 0; j < 8; j++) acc = __fmaf_rn(__bfloat162float(A[j]), __bfloat162float(B[j]), acc);
    return acc;
}
// Packed bf16x2 FMA: 4 __hfma2 + one horizontal add, vs 16 converts + 8 FMAs.
__device__ __forceinline__ void dot8_h2(const bf16v8& a, const bf16v8& b, __nv_bfloat162& acc2) {
    const __nv_bfloat162* A = (const __nv_bfloat162*)&a.v;
    const __nv_bfloat162* B = (const __nv_bfloat162*)&b.v;
#pragma unroll
    for (int j = 0; j < 4; j++) acc2 = __hfma2(A[j], B[j], acc2);
}
__device__ __forceinline__ float wsum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
    return v;
}

// ---------------- A: probe-style, no unroll, x from global ----------------
__global__ __launch_bounds__(BLOCK) void gemv_A(__nv_bfloat16* __restrict__ C,
                                                const __nv_bfloat16* __restrict__ x,
                                                const __nv_bfloat16* __restrict__ W, unsigned N,
                                                unsigned K, unsigned nblk) {
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const unsigned nchunk = (K + GV_STEP - 1) / GV_STEP;
    const unsigned per = (N + nblk - 1) / nblk;
    const unsigned n0 = blockIdx.x * per, n1 = (n0 + per < N) ? (n0 + per) : N;
    for (unsigned n = n0 + warp; n < n1; n += WPB) {
        const __nv_bfloat16* wr = W + (size_t)n * K;
        float acc = 0.f;
        for (unsigned c = 0; c < nchunk; c++) {
            unsigned k = c * GV_STEP + lane * 8u;
            if (k < K) acc = dot8(ld8(wr + k), ld8(x + k), acc);
        }
        acc = wsum(acc);
        if (lane == 0) C[n] = __float2bfloat16(acc);
    }
}

// ---------------- B: production gemv_rows — UN loads pre-issued, x global -------------
template <int UN>
__global__ __launch_bounds__(BLOCK) void gemv_B(__nv_bfloat16* __restrict__ C,
                                                const __nv_bfloat16* __restrict__ x,
                                                const __nv_bfloat16* __restrict__ W, unsigned N,
                                                unsigned K, unsigned nblk) {
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const unsigned nchunk = (K + GV_STEP - 1) / GV_STEP;
    const unsigned per = (N + nblk - 1) / nblk;
    const unsigned n0 = blockIdx.x * per, n1 = (n0 + per < N) ? (n0 + per) : N;
    for (unsigned n = n0 + warp; n < n1; n += WPB) {
        const __nv_bfloat16* wr = W + (size_t)n * K;
        float acc = 0.f;
        for (unsigned c = 0; c < nchunk; c += UN) {
            bf16v8 wv[UN];
            unsigned kk[UN];
#pragma unroll
            for (int u = 0; u < UN; u++) {
                const unsigned k = (c + (unsigned)u) * GV_STEP + lane * 8u;
                kk[u] = k;
                wv[u] = (k < K) ? ld8(wr + k) : zero8();
            }
#pragma unroll
            for (int u = 0; u < UN; u++) {
                if (kk[u] >= K) continue;
                acc = dot8(wv[u], ld8(x + kk[u]), acc);
            }
        }
        acc = wsum(acc);
        if (lane == 0) C[n] = __float2bfloat16(acc);
    }
}

// ---------------- C: x staged in smem, UN W loads pre-issued ----------------
template <int UN>
__global__ __launch_bounds__(BLOCK) void gemv_C(__nv_bfloat16* __restrict__ C,
                                                const __nv_bfloat16* __restrict__ x,
                                                const __nv_bfloat16* __restrict__ W, unsigned N,
                                                unsigned K, unsigned nblk) {
    __shared__ __nv_bfloat16 xs[MAXK];
    for (unsigned i = threadIdx.x * 8; i < K; i += BLOCK * 8) *(uint4*)(xs + i) = *(const uint4*)(x + i);
    __syncthreads();

    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const unsigned nchunk = (K + GV_STEP - 1) / GV_STEP;
    const unsigned per = (N + nblk - 1) / nblk;
    const unsigned n0 = blockIdx.x * per, n1 = (n0 + per < N) ? (n0 + per) : N;
    for (unsigned n = n0 + warp; n < n1; n += WPB) {
        const __nv_bfloat16* wr = W + (size_t)n * K;
        float acc = 0.f;
        for (unsigned c = 0; c < nchunk; c += UN) {
            bf16v8 wv[UN];
            unsigned kk[UN];
#pragma unroll
            for (int u = 0; u < UN; u++) {
                const unsigned k = (c + (unsigned)u) * GV_STEP + lane * 8u;
                kk[u] = k;
                wv[u] = (k < K) ? ld8(wr + k) : zero8();
            }
#pragma unroll
            for (int u = 0; u < UN; u++) {
                if (kk[u] >= K) continue;
                acc = dot8(wv[u], ld8(xs + kk[u]), acc);
            }
        }
        acc = wsum(acc);
        if (lane == 0) C[n] = __float2bfloat16(acc);
    }
}

// ---------------- D: warp owns R rows at once, x from smem ----------------
// R independent W streams per warp => R*UN loads in flight, and each x element
// feeds R FMAs (x traffic amortised R-fold).
template <int R, int UN>
__global__ __launch_bounds__(BLOCK) void gemv_D(__nv_bfloat16* __restrict__ C,
                                                const __nv_bfloat16* __restrict__ x,
                                                const __nv_bfloat16* __restrict__ W, unsigned N,
                                                unsigned K, unsigned nblk) {
    __shared__ __nv_bfloat16 xs[MAXK];
    for (unsigned i = threadIdx.x * 8; i < K; i += BLOCK * 8) *(uint4*)(xs + i) = *(const uint4*)(x + i);
    __syncthreads();

    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const unsigned nchunk = (K + GV_STEP - 1) / GV_STEP;
    const unsigned per = (N + nblk - 1) / nblk;
    const unsigned n0 = blockIdx.x * per, n1 = (n0 + per < N) ? (n0 + per) : N;
    // Each warp strides R-row groups.
    for (unsigned nb = n0 + warp * R; nb < n1; nb += WPB * R) {
        float acc[R];
#pragma unroll
        for (int r = 0; r < R; r++) acc[r] = 0.f;
        const unsigned nrows = (nb + R <= n1) ? R : (n1 - nb);
        for (unsigned c = 0; c < nchunk; c += UN) {
            bf16v8 wv[R][UN];
            unsigned kk[UN];
#pragma unroll
            for (int u = 0; u < UN; u++) kk[u] = (c + (unsigned)u) * GV_STEP + lane * 8u;
#pragma unroll
            for (int r = 0; r < R; r++) {
                const __nv_bfloat16* wr = W + (size_t)(nb + r) * K;
#pragma unroll
                for (int u = 0; u < UN; u++)
                    wv[r][u] = (kk[u] < K && (unsigned)r < nrows) ? ld8(wr + kk[u]) : zero8();
            }
#pragma unroll
            for (int u = 0; u < UN; u++) {
                if (kk[u] >= K) continue;
                const bf16v8 xv = ld8(xs + kk[u]);
#pragma unroll
                for (int r = 0; r < R; r++) acc[r] = dot8(wv[r][u], xv, acc[r]);
            }
        }
#pragma unroll
        for (int r = 0; r < R; r++) {
            const float t = wsum(acc[r]);
            if (lane == 0 && (unsigned)r < nrows) C[nb + r] = __float2bfloat16(t);
        }
    }
}

// ---------------- E: D with packed bf16x2 FMA ----------------
template <int R, int UN>
__global__ __launch_bounds__(BLOCK) void gemv_E(__nv_bfloat16* __restrict__ C,
                                                const __nv_bfloat16* __restrict__ x,
                                                const __nv_bfloat16* __restrict__ W, unsigned N,
                                                unsigned K, unsigned nblk) {
    __shared__ __nv_bfloat16 xs[MAXK];
    for (unsigned i = threadIdx.x * 8; i < K; i += BLOCK * 8) *(uint4*)(xs + i) = *(const uint4*)(x + i);
    __syncthreads();

    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const unsigned nchunk = (K + GV_STEP - 1) / GV_STEP;
    const unsigned per = (N + nblk - 1) / nblk;
    const unsigned n0 = blockIdx.x * per, n1 = (n0 + per < N) ? (n0 + per) : N;
    for (unsigned nb = n0 + warp * R; nb < n1; nb += WPB * R) {
        __nv_bfloat162 acc[R];
#pragma unroll
        for (int r = 0; r < R; r++) acc[r] = __float2bfloat162_rn(0.f);
        const unsigned nrows = (nb + R <= n1) ? R : (n1 - nb);
        for (unsigned c = 0; c < nchunk; c += UN) {
            bf16v8 wv[R][UN];
            unsigned kk[UN];
#pragma unroll
            for (int u = 0; u < UN; u++) kk[u] = (c + (unsigned)u) * GV_STEP + lane * 8u;
#pragma unroll
            for (int r = 0; r < R; r++) {
                const __nv_bfloat16* wr = W + (size_t)(nb + r) * K;
#pragma unroll
                for (int u = 0; u < UN; u++)
                    wv[r][u] = (kk[u] < K && (unsigned)r < nrows) ? ld8(wr + kk[u]) : zero8();
            }
#pragma unroll
            for (int u = 0; u < UN; u++) {
                if (kk[u] >= K) continue;
                const bf16v8 xv = ld8(xs + kk[u]);
#pragma unroll
                for (int r = 0; r < R; r++) dot8_h2(wv[r][u], xv, acc[r]);
            }
        }
#pragma unroll
        for (int r = 0; r < R; r++) {
            const float t = wsum(__bfloat162float(acc[r].x) + __bfloat162float(acc[r].y));
            if (lane == 0 && (unsigned)r < nrows) C[nb + r] = __float2bfloat16(t);
        }
    }
}

// ---------------- F: D, but each 8-elem chunk is reduced with 4 packed hfma2 into a
// bf16x2 partial that is then folded into a FLOAT accumulator. Cuts the 24-op dot8 to
// ~7 ops without a long bf16 accumulation chain (only 4 terms ever accumulate in bf16).
template <int R, int UN>
__global__ __launch_bounds__(BLOCK) void gemv_F(__nv_bfloat16* __restrict__ C,
                                                const __nv_bfloat16* __restrict__ x,
                                                const __nv_bfloat16* __restrict__ W, unsigned N,
                                                unsigned K, unsigned nblk) {
    __shared__ __nv_bfloat16 xs[MAXK];
    for (unsigned i = threadIdx.x * 8; i < K; i += BLOCK * 8) *(uint4*)(xs + i) = *(const uint4*)(x + i);
    __syncthreads();

    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const unsigned nchunk = (K + GV_STEP - 1) / GV_STEP;
    const unsigned per = (N + nblk - 1) / nblk;
    const unsigned n0 = blockIdx.x * per, n1 = (n0 + per < N) ? (n0 + per) : N;
    for (unsigned nb = n0 + warp * R; nb < n1; nb += WPB * R) {
        float acc[R];
#pragma unroll
        for (int r = 0; r < R; r++) acc[r] = 0.f;
        const unsigned nrows = (nb + R <= n1) ? R : (n1 - nb);
        for (unsigned c = 0; c < nchunk; c += UN) {
            bf16v8 wv[R][UN];
            unsigned kk[UN];
#pragma unroll
            for (int u = 0; u < UN; u++) kk[u] = (c + (unsigned)u) * GV_STEP + lane * 8u;
#pragma unroll
            for (int r = 0; r < R; r++) {
                const __nv_bfloat16* wr = W + (size_t)(nb + r) * K;
#pragma unroll
                for (int u = 0; u < UN; u++)
                    wv[r][u] = (kk[u] < K && (unsigned)r < nrows) ? ld8(wr + kk[u]) : zero8();
            }
#pragma unroll
            for (int u = 0; u < UN; u++) {
                if (kk[u] >= K) continue;
                const bf16v8 xv = ld8(xs + kk[u]);
#pragma unroll
                for (int r = 0; r < R; r++) {
                    __nv_bfloat162 t = __float2bfloat162_rn(0.f);
                    dot8_h2(wv[r][u], xv, t);
                    acc[r] += __bfloat162float(t.x) + __bfloat162float(t.y);
                }
            }
        }
#pragma unroll
        for (int r = 0; r < R; r++) {
            const float t = wsum(acc[r]);
            if (lane == 0 && (unsigned)r < nrows) C[nb + r] = __float2bfloat16(t);
        }
    }
}

// ------------------------------- harness -------------------------------
// Deterministic, n-dependent fill so the correctness check has real signal.
__global__ void fill_w(__nv_bfloat16* W, size_t n) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        const unsigned h = (unsigned)(i * 2654435761u) >> 24;
        W[i] = __float2bfloat16((float)(h % 17) * 0.0625f - 0.5f);
    }
}
__global__ void fill_x(__nv_bfloat16* x, unsigned K) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < K) x[i] = __float2bfloat16((float)((i * 37) % 13) * 0.03125f - 0.2f);
}

static float* g_flush = nullptr;
static size_t g_flush_n = 256ull << 20;
static void flushL2() { cudaMemsetAsync(g_flush, 0, g_flush_n, 0); }

struct Shape {
    const char* name;
    unsigned N, K;
};

typedef void (*Launch)(dim3, dim3, __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
                       unsigned, unsigned, unsigned);

struct Variant {
    const char* name;
    void (*run)(int nblk, __nv_bfloat16* C, const __nv_bfloat16* x, const __nv_bfloat16* W,
                unsigned N, unsigned K);
};

#define MKRUN(fn, ...)                                                                     \
    [](int nblk, __nv_bfloat16* C, const __nv_bfloat16* x, const __nv_bfloat16* W,         \
       unsigned N, unsigned K) { fn<__VA_ARGS__><<<nblk, BLOCK>>>(C, x, W, N, K, nblk); }
#define MKRUN0(fn)                                                                         \
    [](int nblk, __nv_bfloat16* C, const __nv_bfloat16* x, const __nv_bfloat16* W,         \
       unsigned N, unsigned K) { fn<<<nblk, BLOCK>>>(C, x, W, N, K, nblk); }

int main(int argc, char** argv) {
    cudaDeviceProp p;
    CHK(cudaGetDeviceProperties(&p, 0));
    const int nSM = p.multiProcessorCount;
    printf("# %s SMs=%d  achievable read BW (E1) = %.0f GB/s\n", p.name, nSM, HBM_ACHIEVABLE);

    Shape sh[] = {
        {"qkv_proj (N8192,K2816)", 8192, 2816},
        {"o_proj   (N2816,K4096)", 2816, 4096},
        {"gate_up  (N4224,K2816)", 4224, 2816},
        {"down_prj (N2816,K2112)", 2816, 2112},
        // MoE top-8 experts, flattened over (expert,row) — the schedule op_moe.cuh uses.
        {"moe_gu   (N11264,K2816)", 11264, 2816},
        {"moe_dn   (N22528,K704)", 22528, 704},
        {"lm_head  (N262144,K2816)", 262144, 2816},
    };
    const int NSH = sizeof(sh) / sizeof(sh[0]);

    Variant vs[] = {
        {"A_noun      ", MKRUN0(gemv_A)},
        {"B_un8  (prod)", MKRUN(gemv_B, 8)},
        {"C_smemx_un8 ", MKRUN(gemv_C, 8)},
        {"D_R2_un4    ", MKRUN(gemv_D, 2, 4)},
        {"D_R4_un2    ", MKRUN(gemv_D, 4, 2)},
        {"D_R4_un4    ", MKRUN(gemv_D, 4, 4)},
        {"D_R8_un2    ", MKRUN(gemv_D, 8, 2)},
        {"E_R4_un4_h2 ", MKRUN(gemv_E, 4, 4)},
        {"F_R4_un4_h2f", MKRUN(gemv_F, 4, 4)},
        {"F_R4_un8_h2f", MKRUN(gemv_F, 4, 8)},
        {"F_R8_un4_h2f", MKRUN(gemv_F, 8, 4)},
    };
    const int NV = sizeof(vs) / sizeof(vs[0]);

    CHK(cudaMalloc(&g_flush, g_flush_n));

    const int occs[] = {1, 2, 3};
    const int IT = 20;

    for (int s = 0; s < NSH; s++) {
        const unsigned N = sh[s].N, K = sh[s].K;
        const size_t wbytes = (size_t)N * K * 2;
        __nv_bfloat16 *W, *x, *C, *Cref;
        CHK(cudaMalloc(&W, wbytes));
        CHK(cudaMalloc(&x, K * 2));
        CHK(cudaMalloc(&C, N * 2));
        CHK(cudaMalloc(&Cref, N * 2));
        fill_x<<<(K + 255) / 256, 256>>>(x, K);
        fill_w<<<2048, 256>>>(W, (size_t)N * K);
        CHK(cudaDeviceSynchronize());
        printf("\n== %s   W=%.1f MB ==\n", sh[s].name, wbytes / 1048576.0);
        printf("%-14s", "variant");
        for (int occ : occs) printf("  %d blk/SM: us  GB/s  %%HBM   ", occ);
        printf("\n");

        for (int v = 0; v < NV; v++) {
            printf("%-14s", vs[v].name);
            for (int oi = 0; oi < 3; oi++) {
                const int nblk = nSM * occs[oi];
                for (int i = 0; i < 3; i++) vs[v].run(nblk, C, x, W, N, K);
                CHK(cudaDeviceSynchronize());
                // time with L2 flushed before every launch so weights are always cold
                cudaEvent_t a, b;
                cudaEventCreate(&a);
                cudaEventCreate(&b);
                float total = 0.f;
                for (int i = 0; i < IT; i++) {
                    flushL2();
                    cudaEventRecord(a);
                    vs[v].run(nblk, C, x, W, N, K);
                    cudaEventRecord(b);
                    CHK(cudaDeviceSynchronize());
                    float ms = 0;
                    cudaEventElapsedTime(&ms, a, b);
                    total += ms;
                }
                cudaEventDestroy(a);
                cudaEventDestroy(b);
                const double us = total / IT * 1e3;
                const double bw = wbytes / (us * 1e-6) / 1e9;
                printf("  %7.1f %6.0f %5.1f%%   ", us, bw, 100.0 * bw / HBM_ACHIEVABLE);
                if (v == 0 && oi == 0) CHK(cudaMemcpy(Cref, C, N * 2, cudaMemcpyDeviceToDevice));
            }
            printf("\n");
            // correctness vs variant A
            if (v > 0) {
                std::vector<__nv_bfloat16> hc(N), hr(N);
                CHK(cudaMemcpy(hc.data(), C, N * 2, cudaMemcpyDeviceToHost));
                CHK(cudaMemcpy(hr.data(), Cref, N * 2, cudaMemcpyDeviceToHost));
                double num = 0, den = 0;
                for (unsigned i = 0; i < N; i++) {
                    double d = (double)__bfloat162float(hc[i]) - (double)__bfloat162float(hr[i]);
                    num += d * d;
                    den += (double)__bfloat162float(hr[i]) * (double)__bfloat162float(hr[i]);
                }
                const double rel = den > 0 ? sqrt(num / den) : sqrt(num);
                printf("%-14s  relL2 vs A = %.3e%s\n", "", rel,
                       rel == 0.0 ? "  (BIT-IDENTICAL)" : (rel > 5e-2 ? "  !! TOO LARGE" : ""));
            }
        }
        cudaFree(W);
        cudaFree(x);
        cudaFree(C);
        cudaFree(Cref);
    }
    return 0;
}
