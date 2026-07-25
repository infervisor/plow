/* t3_pipe_probe.cu — prove the cp.async-friendly B operand path for the sm_120 prefill GEMM.
 *
 * QUESTION: the current d_gemm stages the mma B operand (weight, [n][k] in gmem) TRANSPOSED into
 * smem as [k][n] and reads it with ldmatrix.x2.TRANS. That scatter (8 elements written to 8 different
 * smem rows) is incompatible with cp.async, which needs a contiguous 16B gmem->smem copy. This probe
 * proves a drop-in replacement: stage B in its NATURAL [n][k] layout (cp.async-friendly, each n-row is
 * K contiguous bf16) and read the mma B operand with ldmatrix.x2 NON-trans.
 *
 * The two paths must produce BIT-IDENTICAL mma B fragments (and therefore bit-identical f32
 * accumulators), because the mma operands are the same bf16 values in the same lanes. This harness
 * builds one warp's 32x64 output sub-tile (2 m-frag x 8 n-frag, K=32) THREE ways and diffs:
 *   OLD : [k][n] smem + ldmatrix.x2.trans  (the in-tree path)
 *   NEW : [n][k] smem + ldmatrix.x2         (the cp.async-friendly path)
 *   CPU : double-precision reference
 *
 * Build (off-lease):  nvcc -arch=sm_120a -O2 -o /tmp/t3probe runtime/nvidia/experiments/t3_pipe_probe.cu
 * Run   (on-lease):   gpulease t3-pipe /tmp/t3probe
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_bf16.h>

using bf16 = __nv_bfloat16;

#define WM 32
#define WN 64
#define BK 32
#define MFRAG (WM / 16)   /* 2 */
#define NFRAG (WN / 8)    /* 8 */
#define ASTRIDE (BK + 8)  /* 40, k-major A smem */
#define BSTRIDE_OLD (WN + 8)  /* 72, [k][n] smem */
#define NKSTRIDE (BK + 8) /* 40, [n][k] smem, k contiguous */

__device__ __forceinline__ void ldm_x4(unsigned (&r)[4], const void* s) {
    unsigned a = (unsigned)__cvta_generic_to_shared(s);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldm_x2_trans(unsigned (&r)[2], const void* s) {
    unsigned a = (unsigned)__cvta_generic_to_shared(s);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}
__device__ __forceinline__ void ldm_x2(unsigned (&r)[2], const void* s) {
    unsigned a = (unsigned)__cvta_generic_to_shared(s);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}
__device__ __forceinline__ void mma(float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2],
                                    const float (&c)[4]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]),
                   "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

/* A [WM][BK] row-major, B [WN][BK] natural weight [n][k]. C [WM][WN] = A . B^T. use_new picks path. */
template <int USE_NEW>
__global__ void probe(float* C, const bf16* A, const bf16* B) {
    __shared__ bf16 As[WM * ASTRIDE];
    __shared__ bf16 Bold[BK * BSTRIDE_OLD];   /* [k][n] transpose-scatter */
    __shared__ bf16 Bnew[WN * NKSTRIDE];      /* [n][k] natural */
    const int tid = threadIdx.x, lane = tid & 31;

    /* stage A [m][k] */
    for (int i = tid; i < WM * BK; i += 32) {
        int r = i / BK, c = i % BK;
        As[r * ASTRIDE + c] = A[r * BK + c];
    }
    if (USE_NEW) {
        /* natural [n][k] — this is what cp.async will fill from B[n*K+k] directly */
        for (int i = tid; i < WN * BK; i += 32) {
            int n = i / BK, k = i % BK;
            Bnew[n * NKSTRIDE + k] = B[n * BK + k];
        }
    } else {
        /* transpose-scatter [k][n] (the current pgm_stage_b) */
        for (int i = tid; i < WN * BK; i += 32) {
            int n = i / BK, k = i % BK;
            Bold[k * BSTRIDE_OLD + n] = B[n * BK + k];
        }
    }
    __syncthreads();

    float acc[MFRAG][NFRAG][4];
#pragma unroll
    for (int mi = 0; mi < MFRAG; mi++)
        for (int nj = 0; nj < NFRAG; nj++)
            for (int e = 0; e < 4; e++) acc[mi][nj][e] = 0.f;

#pragma unroll
    for (int kf = 0; kf < BK; kf += 16) {
        unsigned af[MFRAG][4];
#pragma unroll
        for (int mi = 0; mi < MFRAG; mi++) {
            int arow = mi * 16 + (lane % 16);
            int acol = kf + (lane / 16) * 8;
            ldm_x4(af[mi], &As[arow * ASTRIDE + acol]);
        }
        unsigned bf[NFRAG][2];
#pragma unroll
        for (int nj = 0; nj < NFRAG; nj++) {
            if (USE_NEW) {
                /* lane l<16 gives the address; n = l&7 (+nj*8), khalf = (l>>3)&1 */
                int n = nj * 8 + (lane & 7);
                int k = kf + ((lane >> 3) & 1) * 8;
                ldm_x2(bf[nj], &Bnew[n * NKSTRIDE + k]);
            } else {
                int brow = kf + (lane % 16);
                ldm_x2_trans(bf[nj], &Bold[brow * BSTRIDE_OLD + nj * 8]);
            }
        }
#pragma unroll
        for (int mi = 0; mi < MFRAG; mi++)
#pragma unroll
            for (int nj = 0; nj < NFRAG; nj++)
                mma(acc[mi][nj], af[mi], bf[nj], acc[mi][nj]);
    }

    /* epilogue: same lane->(row,col) map as d_gemm */
#pragma unroll
    for (int mi = 0; mi < MFRAG; mi++)
#pragma unroll
        for (int nj = 0; nj < NFRAG; nj++) {
            int gr = mi * 16 + (lane / 4);
            int gc = nj * 8 + (lane % 4) * 2;
#pragma unroll
            for (int e = 0; e < 4; e++) {
                int rr = gr + (e / 2) * 8;
                int cc = gc + (e % 2);
                C[rr * WN + cc] = acc[mi][nj][e];
            }
        }
}

int main() {
    std::vector<float> hA(WM * BK), hB(WN * BK);
    unsigned s = 12345u;
    auto rng = [&]() { s = s * 1664525u + 1013904223u; return (float)((int)s) / 2.147483648e9f; };
    std::vector<bf16> bA(WM * BK), bB(WN * BK);
    for (int i = 0; i < WM * BK; i++) { hA[i] = rng(); bA[i] = __float2bfloat16(hA[i]); hA[i] = __bfloat162float(bA[i]); }
    for (int i = 0; i < WN * BK; i++) { hB[i] = rng(); bB[i] = __float2bfloat16(hB[i]); hB[i] = __bfloat162float(bB[i]); }

    bf16 *dA, *dB; float *dCold, *dCnew;
    cudaMalloc(&dA, WM * BK * 2); cudaMalloc(&dB, WN * BK * 2);
    cudaMalloc(&dCold, WM * WN * 4); cudaMalloc(&dCnew, WM * WN * 4);
    cudaMemcpy(dA, bA.data(), WM * BK * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, bB.data(), WN * BK * 2, cudaMemcpyHostToDevice);

    probe<0><<<1, 32>>>(dCold, dA, dB);
    probe<1><<<1, 32>>>(dCnew, dA, dB);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("CUDA ERROR: %s\n", cudaGetErrorString(e)); return 2; }

    std::vector<float> Cold(WM * WN), Cnew(WM * WN);
    cudaMemcpy(Cold.data(), dCold, WM * WN * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(Cnew.data(), dCnew, WM * WN * 4, cudaMemcpyDeviceToHost);

    /* CPU double ref */
    int old_vs_new_mismatch = 0, new_vs_cpu_bad = 0;
    double maxrel = 0;
    for (int i = 0; i < WM; i++)
        for (int j = 0; j < WN; j++) {
            double acc = 0;
            for (int k = 0; k < BK; k++) acc += (double)hA[i * BK + k] * (double)hB[j * BK + k];
            float ref = (float)acc;
            float vn = Cnew[i * WN + j], vo = Cold[i * WN + j];
            if (memcmp(&vn, &vo, 4) != 0) old_vs_new_mismatch++;
            double rel = fabs(vn - ref) / (fabs(ref) + 1e-6);
            if (rel > maxrel) maxrel = rel;
            if (rel > 5e-3) new_vs_cpu_bad++;
        }
    printf("OLD-vs-NEW bit mismatches: %d / %d\n", old_vs_new_mismatch, WM * WN);
    printf("NEW-vs-CPU  max rel err  : %.3e  (bad>%g: %d)\n", maxrel, 5e-3, new_vs_cpu_bad);
    bool ok = (old_vs_new_mismatch == 0) && (new_vs_cpu_bad == 0);
    printf("RESULT: %s\n", ok ? "PASS — [n][k] + ldmatrix.x2 non-trans is bit-exact vs [k][n]+trans"
                              : "FAIL");
    return ok ? 0 : 1;
}
