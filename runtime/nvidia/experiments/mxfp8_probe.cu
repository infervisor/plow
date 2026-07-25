// mxfp8_probe.cu — Does MXFP8 block-scaled tensor-core MMA exist on sm_120a?
//
// GATE for the whole MXFP8 roadmap. The instruction under test is
//   mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X
//                    .f32.e4m3.e4m3.f32.ue8m0  d, a, b, c, {sa},{bidA,tidA}, {sb},{bidB,tidB};
// (e4m3 operands + one UE8M0 power-of-two scale per 32-element block, applied in-MMA).
//
// RESULT: **NO.** ptxas -arch=sm_120a rejects it outright:
//   error: Instruction 'mma with block scale' not supported on .target 'sm_120'
//   error: Feature '.kind::mxf8f6f4' / '.block_scale' / '.scale_vec::1X' not supported on 'sm_120'
// block_scale MMA is a tcgen05 (sm_100a/sm_103a datacenter Blackwell) feature; sm_120
// (consumer/pro Blackwell, GB202/RTX PRO 6000) has warp-level mma.sync ONLY — no in-MMA
// scaling. MXFP8 on this GPU must dequant the UE8M0 block scale in software and feed plain
// e4m3 mma (or FFMA). To keep the block_scale probe reproducible without breaking the build,
// it lives behind #ifdef TRY_BLOCK_SCALE (compile that TU to see the errors above).
//
// This file therefore measures the FALLBACK the roadmap actually has:
//   (1) numeric verify of plain m16n8k32 e4m3 on a 128x128x128 tile vs an f32 CPU reference,
//   (2) achieved TFLOPS of plain m16n8k32 e4m3 (the ceiling MXFP8 compute must live under).
//
// Build: /usr/local/cuda/bin/nvcc -arch=sm_120a -O3 -o /tmp/mx mxfp8_probe.cu
//   (add -DTRY_BLOCK_SCALE to reproduce the ptxas rejection)

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp8.h>

#define CHK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA ERR %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

#ifdef TRY_BLOCK_SCALE
// Compile with -DTRY_BLOCK_SCALE to make ptxas emit the "not supported on sm_120" errors.
__global__ void block_scale_attempt(const uint32_t *A, const uint32_t *B, float *D, uint32_t sa, uint32_t sb) {
  uint32_t a[4] = {A[0], A[1], A[2], A[3]}, b[2] = {B[0], B[1]};
  float d[4] = {0, 0, 0, 0};
  asm volatile(
    "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {0, 0}, {%11}, {0, 0};\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sa), "r"(sb));
  D[threadIdx.x] = d[0];
}
#endif

// ---- plain m16n8k32 e4m3 mma (validated fragment layout, from fp8_verify.cu) ----
//   A: lane L holds rows (L>>2) and (L>>2)+8, k = 8*(L&3)..+7   (row-major [16][32])
//   B: lane L holds col  (L>>2),              k = 8*(L&3)..+7   (col-major [8][32])
//   C: lane L reg r -> row = (L>>2)+8*(r>>1), col = 2*(L&3)+(r&1)
__device__ __forceinline__ void mma_e4m3_16x8x32(float d[4], const uint32_t a[4], const uint32_t b[2]) {
  asm volatile(
    "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// ---------- (1) correctness: C[128][128] = A[128][:]·B[128][:] (B row-major == col-major operand) ----------
// One block, 8 warps. Warp w owns C-rowtile w (16 rows). It sweeps all 16 col-tiles x 4 k-tiles.
#define DIM 128
__global__ void gemm_e4m3_128(const __nv_fp8_e4m3 *A, const __nv_fp8_e4m3 *B, float *C) {
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int rt = warp * 16;                       // this warp's C row-tile base
  for (int ct = 0; ct < DIM; ct += 8) {     // 16 col-tiles
    float d[4] = {0, 0, 0, 0};
    for (int kt = 0; kt < DIM; kt += 32) {  // 4 k-tiles
      int rlo = rt + (lane >> 2), rhi = rlo + 8, kb = kt + 8 * (lane & 3);
      uint32_t a[4], b[2];
      a[0] = *(const uint32_t *)(A + rlo * DIM + kb);
      a[2] = *(const uint32_t *)(A + rlo * DIM + kb + 4);
      a[1] = *(const uint32_t *)(A + rhi * DIM + kb);
      a[3] = *(const uint32_t *)(A + rhi * DIM + kb + 4);
      int col = ct + (lane >> 2);
      b[0] = *(const uint32_t *)(B + col * DIM + kb);
      b[1] = *(const uint32_t *)(B + col * DIM + kb + 4);
      mma_e4m3_16x8x32(d, a, b);
    }
    for (int r = 0; r < 4; r++) {
      int row = rt + (lane >> 2) + 8 * (r >> 1), cc = ct + 2 * (lane & 3) + (r & 1);
      C[row * DIM + cc] = d[r];
    }
  }
}

// ---------- (2) throughput: tight independent-accumulator mma chains ----------
// Each warp runs NACC independent m16n8k32 accumulators for ITERS steps => hides the
// accumulator RAW latency and saturates the tensor pipe. Flops = 2*16*8*32 per mma.
#define NACC 8
__global__ void bench_e4m3(const uint32_t *A, const uint32_t *B, float *sink, int iters) {
  int lane = threadIdx.x & 31;
  uint32_t a[4] = {A[lane], A[lane + 32], A[lane + 64], A[lane + 96]};
  uint32_t b[2] = {B[lane], B[lane + 32]};
  float d[NACC][4];
  #pragma unroll
  for (int i = 0; i < NACC; i++) { d[i][0] = lane; d[i][1] = 0; d[i][2] = 0; d[i][3] = 0; }
  for (int t = 0; t < iters; t++) {
    #pragma unroll
    for (int i = 0; i < NACC; i++) mma_e4m3_16x8x32(d[i], a, b);
  }
  float s = 0;
  #pragma unroll
  for (int i = 0; i < NACC; i++) s += d[i][0] + d[i][1] + d[i][2] + d[i][3];
  if (s == -123.f) sink[threadIdx.x] = s;
}

int main() {
  cudaDeviceProp prop; CHK(cudaGetDeviceProperties(&prop, 0));
  const int grid = prop.multiProcessorCount;
  printf("# %s  SMs=%d\n", prop.name, grid);
  printf("# MXFP8 block_scale MMA: NOT SUPPORTED on sm_120 (ptxas rejects .kind::mxf8f6f4/.block_scale).\n");
  printf("# Measuring the plain e4m3 mma.sync fallback the roadmap must use.\n\n");

  // ---- (1) verify ----
  {
    int NE = DIM * DIM;
    std::vector<float> hA(NE), hB(NE), ref(NE);
    std::vector<__nv_fp8_e4m3> qA(NE), qB(NE);
    srand(1234);
    for (int i = 0; i < NE; i++) { hA[i] = (float)(rand() % 7 - 3); qA[i] = __nv_fp8_e4m3(hA[i]); hA[i] = (float)qA[i]; }
    for (int i = 0; i < NE; i++) { hB[i] = (float)(rand() % 7 - 3); qB[i] = __nv_fp8_e4m3(hB[i]); hB[i] = (float)qB[i]; }
    for (int m = 0; m < DIM; m++) for (int n = 0; n < DIM; n++) {
      float s = 0; for (int k = 0; k < DIM; k++) s += hA[m * DIM + k] * hB[n * DIM + k];
      ref[m * DIM + n] = s;
    }
    __nv_fp8_e4m3 *dA, *dB; float *dC;
    CHK(cudaMalloc(&dA, NE)); CHK(cudaMalloc(&dB, NE)); CHK(cudaMalloc(&dC, NE * 4));
    CHK(cudaMemcpy(dA, qA.data(), NE, cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(dB, qB.data(), NE, cudaMemcpyHostToDevice));
    gemm_e4m3_128<<<1, 256>>>(dA, dB, dC);
    CHK(cudaDeviceSynchronize());
    std::vector<float> hC(NE);
    CHK(cudaMemcpy(hC.data(), dC, NE * 4, cudaMemcpyDeviceToHost));
    int bad = 0; float worst = 0;
    for (int i = 0; i < NE; i++) { float e = fabsf(hC[i] - ref[i]); if (e > worst) worst = e; if (e != 0.f) bad++; }
    printf("(1) e4m3 128x128x128 verify: exact-mismatches=%d/%d  max|err|=%g  %s\n",
           bad, NE, worst, bad ? "FAIL" : "PASS (bit-exact)");
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
  }

  // ---- (2) rate ----
  {
    uint32_t *dA, *dB; float *sink;
    CHK(cudaMalloc(&dA, 128 * 4)); CHK(cudaMalloc(&dB, 64 * 4)); CHK(cudaMalloc(&sink, 256 * 4));
    CHK(cudaMemset(dA, 0x11, 128 * 4)); CHK(cudaMemset(dB, 0x22, 64 * 4));
    int warps_per_sm = 48;                 // saturate the tensor units
    int blk = 256, blocks = grid * (warps_per_sm * 32 / blk);
    int ITERS = 4000;
    bench_e4m3<<<blocks, blk>>>(dA, dB, sink, ITERS);
    CHK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1; CHK(cudaEventCreate(&e0)); CHK(cudaEventCreate(&e1));
    int REP = 20;
    CHK(cudaEventRecord(e0));
    for (int i = 0; i < REP; i++) bench_e4m3<<<blocks, blk>>>(dA, dB, sink, ITERS);
    CHK(cudaEventRecord(e1)); CHK(cudaDeviceSynchronize());
    float ms; CHK(cudaEventElapsedTime(&ms, e0, e1)); ms /= REP;
    double warps = (double)blocks * (blk / 32);
    double mmas = warps * (double)NACC * ITERS;
    double flops = mmas * 2.0 * 16 * 8 * 32;
    printf("(2) e4m3 m16n8k32 rate: %.3f ms/rep  %.0f TFLOPS  (%d blocks x %d thr, NACC=%d, ITERS=%d)\n",
           ms, flops / (ms * 1e-3) / 1e12, blocks, blk, NACC, ITERS);
    cudaFree(dA); cudaFree(dB); cudaFree(sink);
  }
  return 0;
}
