// Batch-1 fp8 weight-only GEMV on the REAL Gemma layer-0 q_proj tensor
// W: [N=8192][K=5376] e4m3 row-major, weight_scale: f32[N] (per output channel).
// Contract: W_bf16 ~= W8 * scale[n]; scale factors out of the K-reduction and is
// applied ONCE in the epilogue after the cross-lane reduction.
//
// Two implementations, identical byte traffic (44 MiB of weights):
//   (a) FFMA  : fp8 dequantized on load, x stays f32
//   (b) MMA   : mma.sync.aligned.m16n8k32 with x quantized to fp8, M=1 of 16 used
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cuda_fp8.h>

#define N 8192
#define K 5376
#define BLOCKS 1024        // 8 output rows per block
#define THREADS 256        // 8 warps

__device__ __forceinline__ float2 fp8x2_to_float2(uint16_t v) {
  __half2_raw h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)v, __NV_E4M3);
  return __half22float2(*reinterpret_cast<__half2 *>(&h));
}

// ---------- (a) FFMA, dequant-on-load ----------
__global__ __launch_bounds__(THREADS) void gemv_ffma(
    const uint8_t *__restrict__ W, const float *__restrict__ x,
    const float *__restrict__ scale, float *__restrict__ y) {
  extern __shared__ float xs[];
  for (int i = threadIdx.x; i < K; i += THREADS) xs[i] = x[i];
  __syncthreads();

  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int n = blockIdx.x * 8 + warp;
  const uint8_t *row = W + (size_t)n * K;
  float acc = 0.f;
  // K = 5376 = 21 steps * 32 lanes * 8 bytes
#pragma unroll 1
  for (int step = 0; step < 21; ++step) {
    int off = step * 256 + lane * 8;
    uint2 w = *(const uint2 *)(row + off);
    const uint16_t *wp = (const uint16_t *)&w;
    // vector shared loads: 2 x float4 instead of 8 scalar reads of x
    float4 x0 = *(const float4 *)(xs + off);
    float4 x1 = *(const float4 *)(xs + off + 4);
    const float *xv0 = (const float *)&x0, *xv1 = (const float *)&x1;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float2 f = fp8x2_to_float2(wp[j]);
      const float *xv = (j < 2) ? xv0 : xv1;
      int b = (2 * j) & 3;
      acc = fmaf(f.x, xv[b], acc);
      acc = fmaf(f.y, xv[b + 1], acc);
    }
  }
  for (int o = 16; o; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
  if (lane == 0) y[n] = acc * scale[n];   // scale applied ONCE, in epilogue
}

// ---------- (a2) FFMA, no shared prologue: x read straight from global ----------
// Isolates the cost of staging all 21.5 KB of x into shared per block.
__global__ __launch_bounds__(THREADS) void gemv_ffma_nosmem(
    const uint8_t *__restrict__ W, const float *__restrict__ x,
    const float *__restrict__ scale, float *__restrict__ y) {
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int n = blockIdx.x * 8 + warp;
  const uint8_t *row = W + (size_t)n * K;
  float acc = 0.f;
#pragma unroll 1
  for (int step = 0; step < 21; ++step) {
    int off = step * 256 + lane * 8;
    uint2 w = *(const uint2 *)(row + off);
    const uint16_t *wp = (const uint16_t *)&w;
    float4 x0 = *(const float4 *)(x + off);
    float4 x1 = *(const float4 *)(x + off + 4);
    const float *xv0 = (const float *)&x0, *xv1 = (const float *)&x1;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float2 f = fp8x2_to_float2(wp[j]);
      const float *xv = (j < 2) ? xv0 : xv1;
      int b = (2 * j) & 3;
      acc = fmaf(f.x, xv[b], acc);
      acc = fmaf(f.y, xv[b + 1], acc);
    }
  }
  for (int o = 16; o; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
  if (lane == 0) y[n] = acc * scale[n];
}

// ---------- (b) mma.sync fp8 ----------
// Warp holds 8 output channels as the mma N dimension. B fragment wants
// col=lane>>2, k=8*(lane&3)+e  -> exactly 8 contiguous bytes of W row n. A holds
// x in mma row 0 (rows 1..15 unused: M=1 uses 1/16 of the tensor-core math).
__global__ __launch_bounds__(THREADS) void gemv_mma(
    const uint8_t *__restrict__ W, const uint8_t *__restrict__ x8,
    const float *__restrict__ scale, float *__restrict__ y) {
  __shared__ float part[8][8];
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int nbase = blockIdx.x * 8;
  int n = nbase + (lane >> 2);
  const uint8_t *row = W + (size_t)n * K;

  int kchunk = K / 8;                  // 672 k per warp
  int k0 = warp * kchunk;
  float d[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 1
  for (int k = k0; k < k0 + kchunk; k += 32) {
    int kk = k + 8 * (lane & 3);
    uint32_t b[2];
    b[0] = *(const uint32_t *)(row + kk);
    b[1] = *(const uint32_t *)(row + kk + 4);
    uint32_t a[4];
    a[0] = *(const uint32_t *)(x8 + kk);        // mma row 0
    a[2] = *(const uint32_t *)(x8 + kk + 4);
    a[1] = 0; a[3] = 0;                          // mma row 8 unused
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  }
  // mma row 0 lives in regs d0,d1 of lanes 0..3: col = 2*(lane&3)+(reg&1)
  if (lane < 4) { part[warp][2 * lane] = d[0]; part[warp][2 * lane + 1] = d[1]; }
  __syncthreads();
  if (threadIdx.x < 8) {
    float s = 0;
    for (int w = 0; w < 8; ++w) s += part[w][threadIdx.x];
    y[nbase + threadIdx.x] = s * scale[nbase + threadIdx.x];
  }
}

// ---------- (c) mma.sync fp8, shared-memory staged ----------
// Fair-shot variant: the direct B-fragment access pattern spreads a warp across
// 8 rows at 32B granularity. Here W is staged into shared with FULLY COALESCED
// global loads (warp w reads 256 contiguous bytes of row w), then the mma reads
// its fragment out of shared where the scatter is free.
#define TILE_K 256          // 5376 = 21 * 256
__global__ __launch_bounds__(THREADS) void gemv_mma_smem(
    const uint8_t *__restrict__ W, const uint8_t *__restrict__ x8,
    const float *__restrict__ scale, float *__restrict__ y) {
  const int STRIDE = TILE_K + 8;                 // pad (8B-aligned, uint2 stores)
  __shared__ uint8_t ws[8 * (TILE_K + 8)];
  __shared__ float part[8][8];
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int nbase = blockIdx.x * 8;
  float d[4] = {0.f, 0.f, 0.f, 0.f};

  for (int t = 0; t < K / TILE_K; ++t) {
    // coalesced stage: warp w loads 256 contiguous bytes of row (nbase+w)
    *(uint2 *)(ws + warp * STRIDE + lane * 8) =
        *(const uint2 *)(W + (size_t)(nbase + warp) * K + t * TILE_K + lane * 8);
    __syncthreads();
    // warp w does the mma for k-chunk [w*32, w*32+32) of this tile
    int kk = warp * 32 + 8 * (lane & 3);
    uint32_t b[2], a[4];
    b[0] = *(const uint32_t *)(ws + (lane >> 2) * STRIDE + kk);
    b[1] = *(const uint32_t *)(ws + (lane >> 2) * STRIDE + kk + 4);
    int gk = t * TILE_K + kk;
    a[0] = *(const uint32_t *)(x8 + gk);
    a[2] = *(const uint32_t *)(x8 + gk + 4);
    a[1] = 0; a[3] = 0;
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    __syncthreads();
  }
  if (lane < 4) { part[warp][2 * lane] = d[0]; part[warp][2 * lane + 1] = d[1]; }
  __syncthreads();
  if (threadIdx.x < 8) {
    float s = 0;
    for (int w = 0; w < 8; ++w) s += part[w][threadIdx.x];
    y[nbase + threadIdx.x] = s * scale[nbase + threadIdx.x];
  }
}

int main() {
  size_t wb = (size_t)N * K;
  uint8_t *hW = (uint8_t *)malloc(wb);
  float *hS = (float *)malloc(N * 4);
  FILE *f = fopen("/workspace/fp8poc/w.bin", "rb");
  if (!f || fread(hW, 1, wb, f) != wb) { printf("w.bin read failed\n"); return 1; }
  fclose(f);
  f = fopen("/workspace/fp8poc/scale.bin", "rb");
  if (!f || fread(hS, 4, N, f) != N) { printf("scale.bin read failed\n"); return 1; }
  fclose(f);

  float *hx = (float *)malloc(K * 4);
  uint8_t *hx8 = (uint8_t *)malloc(K);
  srand(7);
  for (int i = 0; i < K; ++i) {
    hx[i] = (float)(rand() % 2001 - 1000) / 1000.0f;      // ~U(-1,1)
    __nv_fp8_e4m3 q(hx[i]); hx8[i] = *(uint8_t *)&q;
  }

  uint8_t *dW, *dx8; float *dx, *dS, *dy;
  cudaMalloc(&dW, wb); cudaMalloc(&dx, K * 4); cudaMalloc(&dx8, K);
  cudaMalloc(&dS, N * 4); cudaMalloc(&dy, N * 4);
  cudaMemcpy(dW, hW, wb, cudaMemcpyHostToDevice);
  cudaMemcpy(dx, hx, K * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dx8, hx8, K, cudaMemcpyHostToDevice);
  cudaMemcpy(dS, hS, N * 4, cudaMemcpyHostToDevice);

  // CPU reference in f64 over the real weights (first 256 rows only, for speed)
  int RN = 256;
  double *ref = (double *)malloc(RN * sizeof(double));
  for (int n = 0; n < RN; ++n) {
    double s = 0;
    for (int k = 0; k < K; ++k) { __nv_fp8_e4m3 q; *(uint8_t *)&q = hW[(size_t)n * K + k];
      s += (double)(float)q * (double)hx[k]; }
    ref[n] = s * (double)hS[n];
  }

  float *hy = (float *)malloc(N * 4);
  auto check = [&](const char *nm) {
    cudaMemcpy(hy, dy, N * 4, cudaMemcpyDeviceToHost);
    double worst = 0, den = 0;
    for (int n = 0; n < RN; ++n) { worst = fmax(worst, fabs(hy[n] - ref[n])); den = fmax(den, fabs(ref[n])); }
    printf("   %-6s max rel err vs f64 ref (first %d rows) = %.3e\n", nm, RN, worst / den);
  };

  // The 42 MiB tensor fits in the 5090's ~96 MB L2, so a naive repeat loop
  // measures L2, not HBM. Replicate it COPIES times (>L2) and cycle, which is
  // the real decode condition: weights streamed from HBM, never resident.
  const int COPIES = 6;                       // 6 * 42 MiB = 252 MiB >> 96 MB L2
  uint8_t *dWc[COPIES];
  dWc[0] = dW;
  for (int c = 1; c < COPIES; ++c) {
    cudaMalloc(&dWc[c], wb);
    cudaMemcpy(dWc[c], dW, wb, cudaMemcpyDeviceToDevice);
  }
  if (cudaGetLastError() != cudaSuccess) { printf("replica alloc failed\n"); return 1; }

  cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
  int REP = 200;
  size_t shmem = K * sizeof(float);

  gemv_ffma<<<BLOCKS, THREADS, shmem>>>(dW, dx, dS, dy);
  cudaDeviceSynchronize();
  if (cudaGetLastError() != cudaSuccess) { printf("ffma launch err\n"); return 1; }
  check("FFMA");
  cudaEventRecord(s);
  for (int i = 0; i < REP; ++i) gemv_ffma<<<BLOCKS, THREADS, shmem>>>(dWc[i % COPIES], dx, dS, dy);
  cudaEventRecord(e); cudaDeviceSynchronize();
  float ms; cudaEventElapsedTime(&ms, s, e); ms /= REP;
  printf("FFMA dequant-on-load : %7.3f ms   %7.1f GB/s\n", ms, wb / (ms * 1e-3) / 1e9);

  gemv_ffma_nosmem<<<BLOCKS, THREADS>>>(dW, dx, dS, dy);
  cudaDeviceSynchronize();
  if (cudaGetLastError() != cudaSuccess) { printf("ffma_nosmem launch err\n"); return 1; }
  check("FFMA2");
  cudaEventRecord(s);
  for (int i = 0; i < REP; ++i) gemv_ffma_nosmem<<<BLOCKS, THREADS>>>(dWc[i % COPIES], dx, dS, dy);
  cudaEventRecord(e); cudaDeviceSynchronize();
  cudaEventElapsedTime(&ms, s, e); ms /= REP;
  printf("FFMA no-smem x       : %7.3f ms   %7.1f GB/s\n", ms, wb / (ms * 1e-3) / 1e9);

  gemv_mma<<<BLOCKS, THREADS>>>(dW, dx8, dS, dy);
  cudaDeviceSynchronize();
  if (cudaGetLastError() != cudaSuccess) { printf("mma launch err\n"); return 1; }
  check("MMA");
  cudaEventRecord(s);
  for (int i = 0; i < REP; ++i) gemv_mma<<<BLOCKS, THREADS>>>(dWc[i % COPIES], dx8, dS, dy);
  cudaEventRecord(e); cudaDeviceSynchronize();
  cudaEventElapsedTime(&ms, s, e); ms /= REP;
  printf("MMA  m16n8k32 (M=1)  : %7.3f ms   %7.1f GB/s\n", ms, wb / (ms * 1e-3) / 1e9);

  gemv_mma_smem<<<BLOCKS, THREADS>>>(dW, dx8, dS, dy);
  cudaDeviceSynchronize();
  if (cudaGetLastError() != cudaSuccess) { printf("mma_smem launch err: %s\n", cudaGetErrorString(cudaGetLastError())); return 1; }
  check("MMAs");
  cudaEventRecord(s);
  for (int i = 0; i < REP; ++i) gemv_mma_smem<<<BLOCKS, THREADS>>>(dWc[i % COPIES], dx8, dS, dy);
  cudaEventRecord(e); cudaDeviceSynchronize();
  cudaEventElapsedTime(&ms, s, e); ms /= REP;
  printf("MMA  + smem staging  : %7.3f ms   %7.1f GB/s\n", ms, wb / (ms * 1e-3) / 1e9);
  return 0;
}
