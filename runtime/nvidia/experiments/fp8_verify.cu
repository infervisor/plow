// End-to-end verification of the EMPIRICALLY DERIVED fragment layouts.
// Uses small-integer operands so every product and the K-sum are exactly
// representable in f32 -> we can demand a BIT-EXACT match against the CPU
// reference. Any layout error shows up immediately.
//
// Derived layout (k labelled so both operands load contiguously):
//   C  : lane L, reg r  -> row = (L>>2) + 8*(r>>1),  col = 2*(L&3) + (r&1)
//   fp8 m16n8k32  A: lane L holds rows (L>>2) and (L>>2)+8, k = 8*(L&3) .. +7
//                 B: lane L holds col  (L>>2),          k = 8*(L&3) .. +7
//   bf16 m16n8k16 A: lane L holds rows (L>>2) and (L>>2)+8, k = 4*(L&3) .. +3
//                 B: lane L holds col  (L>>2),          k = 4*(L&3) .. +3

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cuda_fp8.h>
#include <cuda_bf16.h>

#ifndef BREAK_LAYOUT
#define BREAK_LAYOUT 0
#endif

// ---------------- fp8 (e4m3 / e5m2) m16n8k32 ----------------
template <typename T8>
__global__ void mma_fp8_k32(const T8 *__restrict__ A,   // [16][32] row-major
                            const T8 *__restrict__ B,   // [8][32]  (col-major B == N x K)
                            float *__restrict__ C) {    // [16][8]
  int L = threadIdx.x;
  int rlo = L >> 2, rhi = rlo + 8, kb = 8 * (L & 3);
#if BREAK_LAYOUT
  kb = 8 * ((L & 3) ^ 1);          // deliberately wrong k mapping for A only
#endif
  uint32_t a[4], b[2];
  a[0] = *(const uint32_t *)(A + rlo * 32 + kb);
  a[2] = *(const uint32_t *)(A + rlo * 32 + kb + 4);
  a[1] = *(const uint32_t *)(A + rhi * 32 + kb);
  a[3] = *(const uint32_t *)(A + rhi * 32 + kb + 4);
  int col = L >> 2, kbb = 8 * (L & 3);
  b[0] = *(const uint32_t *)(B + col * 32 + kbb);
  b[1] = *(const uint32_t *)(B + col * 32 + kbb + 4);

  float d[4] = {0.f, 0.f, 0.f, 0.f};
  asm volatile(
#if E5M2
      "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 "
#else
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
#endif
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));

  for (int r = 0; r < 4; ++r) {
    int row = (L >> 2) + 8 * (r >> 1), cc = 2 * (L & 3) + (r & 1);
    C[row * 8 + cc] = d[r];
  }
}

// ---------------- bf16 m16n8k16 ----------------
__global__ void mma_bf16_k16(const __nv_bfloat16 *__restrict__ A,  // [16][16]
                             const __nv_bfloat16 *__restrict__ B,  // [8][16]
                             float *__restrict__ C) {
  int L = threadIdx.x;
  int rlo = L >> 2, rhi = rlo + 8, kb = 4 * (L & 3);
  uint32_t a[4], b[2];
  a[0] = *(const uint32_t *)(A + rlo * 16 + kb);
  a[2] = *(const uint32_t *)(A + rlo * 16 + kb + 2);
  a[1] = *(const uint32_t *)(A + rhi * 16 + kb);
  a[3] = *(const uint32_t *)(A + rhi * 16 + kb + 2);
  int col = L >> 2;
  b[0] = *(const uint32_t *)(B + col * 16 + kb);
  b[1] = *(const uint32_t *)(B + col * 16 + kb + 2);
  float d[4] = {0.f, 0.f, 0.f, 0.f};
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  for (int r = 0; r < 4; ++r)
    C[((L >> 2) + 8 * (r >> 1)) * 8 + 2 * (L & 3) + (r & 1)] = d[r];
}

template <typename T8>
int run_fp8(const char *name) {
  const int M = 16, N = 8, K = 32;
  float hA[M * K], hB[N * K], ref[M * N];
  T8 *dA, *dB; float *dC, hC[M * N];
  cudaMalloc(&dA, M * K); cudaMalloc(&dB, N * K); cudaMalloc(&dC, M * N * 4);
  T8 *qA = (T8 *)malloc(M * K), *qB = (T8 *)malloc(N * K);
  srand(1234);
  for (int i = 0; i < M * K; ++i) { hA[i] = (float)(rand() % 9 - 4); qA[i] = T8(hA[i]); hA[i] = (float)qA[i]; }
  for (int i = 0; i < N * K; ++i) { hB[i] = (float)(rand() % 9 - 4); qB[i] = T8(hB[i]); hB[i] = (float)qB[i]; }
  for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) {
    float s = 0; for (int k = 0; k < K; ++k) s += hA[m * K + k] * hB[n * K + k];
    ref[m * N + n] = s;
  }
  cudaMemcpy(dA, qA, M * K, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, qB, N * K, cudaMemcpyHostToDevice);
  mma_fp8_k32<T8><<<1, 32>>>(dA, dB, dC);
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { printf("%s CUDA ERROR: %s\n", name, cudaGetErrorString(e)); return 1; }
  cudaMemcpy(hC, dC, M * N * 4, cudaMemcpyDeviceToHost);
  int bad = 0; float worst = 0;
  for (int i = 0; i < M * N; ++i) { float d = fabsf(hC[i] - ref[i]); if (d > worst) worst = d; if (d != 0.f) bad++; }
  printf("%-22s exact-mismatches=%3d / %d   max|err|=%g   %s\n", name, bad, M * N, worst, bad ? "FAIL" : "PASS");
  if (bad) { printf("   gpu[0..7]:"); for (int i=0;i<8;++i) printf(" %6.0f", hC[i]);
             printf("\n   ref[0..7]:"); for (int i=0;i<8;++i) printf(" %6.0f", ref[i]); printf("\n"); }
  return bad != 0;
}

int run_bf16() {
  const int M = 16, N = 8, K = 16;
  float hA[M * K], hB[N * K], ref[M * N], hC[M * N];
  __nv_bfloat16 *dA, *dB, *qA = (__nv_bfloat16 *)malloc(M * K * 2), *qB = (__nv_bfloat16 *)malloc(N * K * 2);
  float *dC;
  cudaMalloc(&dA, M * K * 2); cudaMalloc(&dB, N * K * 2); cudaMalloc(&dC, M * N * 4);
  srand(99);
  for (int i = 0; i < M * K; ++i) { hA[i] = (float)(rand() % 9 - 4); qA[i] = __nv_bfloat16(hA[i]); }
  for (int i = 0; i < N * K; ++i) { hB[i] = (float)(rand() % 9 - 4); qB[i] = __nv_bfloat16(hB[i]); }
  for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) {
    float s = 0; for (int k = 0; k < K; ++k) s += hA[m * K + k] * hB[n * K + k];
    ref[m * N + n] = s;
  }
  cudaMemcpy(dA, qA, M * K * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, qB, N * K * 2, cudaMemcpyHostToDevice);
  mma_bf16_k16<<<1, 32>>>(dA, dB, dC);
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { printf("bf16 CUDA ERROR: %s\n", cudaGetErrorString(e)); return 1; }
  cudaMemcpy(hC, dC, M * N * 4, cudaMemcpyDeviceToHost);
  int bad = 0; float worst = 0;
  for (int i = 0; i < M * N; ++i) { float d = fabsf(hC[i] - ref[i]); if (d > worst) worst = d; if (d != 0.f) bad++; }
  printf("%-22s exact-mismatches=%3d / %d   max|err|=%g   %s\n", "bf16 m16n8k16", bad, M * N, worst, bad ? "FAIL" : "PASS");
  return bad != 0;
}

int main() {
  int f = 0;
#if E5M2
  f |= run_fp8<__nv_fp8_e5m2>("e5m2 m16n8k32");
#else
  f |= run_fp8<__nv_fp8_e4m3>("e4m3 m16n8k32");
#endif
  f |= run_bf16();
  return f;
}
