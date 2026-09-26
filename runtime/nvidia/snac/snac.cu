// libplow_snac.so: native SNAC-24kHz decoder (codes -> 24 kHz PCM) for plowrt.
//
// C ABI (all functions return 0 on success, negative on error):
//   int plow_snac_create(int device, const char* weights_path, int max_batch,
//                        int max_frames, void** out);
//   int plow_snac_destroy(void* h);
//   int plow_snac_decode_host(void* h, const int32_t* codes_host /*[B][F][7]*/, int B, int F,
//                             float* pcm_host /*[B][F*2048]*/, unsigned long long noise_seed);
//       (blocking; internal stream + pinned staging; callable from a thread with no CUDA state)
//   int plow_snac_decode(void* h, const int32_t* codes /*dev [B][F][7]*/, int B, int F,
//                        float* pcm_out /*dev [B][F*2048]*/, unsigned long long noise_seed,
//                        void* stream /*CUstream*/);
// Error codes: -1 bad argument, -2 CUDA error, -4 cannot read weights, -5 bad weight file,
//              -6 B/F exceed the create-time maximum.
//
// Weights: the blob written by scripts/tts/snac_prep.py (format v2, documented there).
// Codes: Orpheus/Veena 7-code frames of raw codebook ids (clamped to 0..4095):
//   level0[f]=c0, level1[2f..2f+1]=c1,c4, level2[4f..4f+3]=c2,c3,c5,c6.
// Output: pcm_out[b][t], t < F*2048, tanh-bounded, 24 kHz.
// noise_seed == 0 disables the NoiseBlocks; otherwise noise[blk,b,t] ~ N(0,1) from a
// splitmix64 hash of (seed, block, b, t) (t local to this call) via Box-Muller.
//
// Runtime API on the device's primary context (what plowrt retains). decode() only
// enqueues on `stream`: no allocation, no sync. Exception: the first call for each
// (B, F, noise on/off) captures + instantiates a CUDA graph (host-side cost, cached for
// the handle's lifetime); PLOW_SNAC_GRAPH=0 at create launches kernels directly instead.
// One handle = one stream at a time (scratch is shared); not thread-safe.
//
// GEMMs (every 1x1 conv and the ConvTransposes) run on a persistent, warp-specialized Hopper
// kernel (TMA loads -> producer warpgroup rounds/splits A in smem -> 2 consumer warpgroups
// issue wgmma TF32, fp32 accumulate; deterministic split-K for small M). No cuBLAS.
// PLOW_SNAC_PREC at create selects:
//   "3xtf32" (default): split operands a = a_hi + a_lo, b = b_hi + b_lo (RN-rounded), three
//            MMAs (lo*hi + hi*lo + hi*hi), each 32-wide K chunk folded into the fp32
//            accumulator with a rounded add -> fp32-level accuracy (rel-L2 ~5e-6 vs PyTorch
//            fp32; plain 1-pass TF32 measures ~1.1e-3).
//   "tf32":  one MMA per k-step (~15% faster end to end, rel-L2 ~1e-3).
//
// Activations are channels-last [B][T][C] fp32. Per DecoderBlock:
//   ConvT : ONE GEMM over all output phases (N = s*Cout, K = 2*Cin: input rows q and q-1);
//           its input is stored [B][T+1][C] with a zero row closing each batch item, so both
//           K halves are plain TMA boxes; snake applied while staging A; bias epilogue
//           scatters rows (no col2im pass)
//   Noise : x2 = x + n[b,t] * (x @ Wn^T) as a GEMM epilogue       (only when seed != 0)
//   RU x3 : k_dw (snake -> depthwise k7 dilated conv -> snake) then GEMM with
//           bias + residual epilogue
// Launches per decode (direct mode): 1 quant/dwconv + 1 pre 1x1 + 4 x (ConvT [+ noise]
// + 3 x (dw + 1x1)) + 1 final = 31 without noise, 35 with (PDL-chained; PLOW_SNAC_PDL=0
// disables). Graph mode (default): 1 cudaGraphLaunch per decode.
// Env knobs are read at create; PLOW_SNAC_DEBUG_STOP / plow_snac_debug_read are test hooks.
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace {

constexpr int kLat = 768, kD0 = 1024, kCodes = 4096, kFrameSamples = 2048;
constexpr int kStride[4] = {8, 8, 4, 2};
constexpr int kDil[3] = {1, 3, 9};
constexpr int kSMs = 132;

// sin(y)^2 has period pi: reduce y by pi (two-constant Cody-Waite) to [-pi/2, pi/2], then
// the SFU sine (abs err ~2^-21 there).
__device__ __forceinline__ float sin2(float y) {
  float k = rintf(y * 0.318309886183790672f);
  float r = fmaf(k, -3.14159274101257324f, y);
  r = fmaf(k, 8.74227800037248e-08f, r);
  float s = __sinf(r);
  return s * s;
}
__device__ __forceinline__ float snake(float x, float alpha, float inv) {
  return fmaf(inv, sin2(alpha * x), x);
}
__device__ __forceinline__ float snake_inv(float alpha) { return 1.0f / (alpha + 1e-9f); }

__device__ __forceinline__ unsigned long long mix64(unsigned long long z) {
  z += 0x9e3779b97f4a7c15ULL;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

__device__ __forceinline__ float noise_at(unsigned long long seed, int blk, int b, int t) {
  unsigned long long key = ((unsigned long long)blk << 58) | ((unsigned long long)b << 32) |
                           (unsigned)t;
  unsigned long long h1 = mix64(seed ^ mix64(key));
  unsigned long long h2 = mix64(h1);
  float u1 = (float)((h1 >> 40) + 1) * (1.0f / 16777216.0f);  // (0,1]
  float u2 = (float)(h2 >> 40) * (1.0f / 16777216.0f);        // [0,1)
  return sqrtf(-2.0f * logf(u1)) * cospif(2.0f * u2);
}

// Programmatic dependent launch: wait for the previous grid's completion + memory flush;
// allow the next grid to start its prologue.
__device__ __forceinline__ void griddep_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }
__device__ __forceinline__ void griddep_launch_dependents() {
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
}

__device__ __forceinline__ float tf32r(float x) {
  uint32_t u;
  asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(u) : "f"(x));
  return __uint_as_float(u);
}

// ---------------------------------------------------------------------------------------
// quantizer.from_codes + the decoder's first depthwise k7 conv (zero padded), fused.
struct QuantArgs {
  const int32_t* codes;
  int B, F;
  const float* P0;
  const float* P1;
  const float* P2;
  const float* dw_w;  // [768][7]
  const float* dw_b;
  float* z;  // [B*4F][768]
  unsigned long long seed;
  unsigned long long* seed_slot;
};

__device__ __forceinline__ int clampc(int c) { return c < 0 ? 0 : (c >= kCodes ? kCodes - 1 : c); }

__global__ void k_quant(QuantArgs a) {
  const int T = 4 * a.F;
  const long n_z = (long)a.B * T * kLat;
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i == 0) *a.seed_slot = a.seed;
  if (i >= n_z) return;
  int c = (int)(i % kLat);
  long bt = i / kLat;
  int t = (int)(bt % T), b = (int)(bt / T);
  const int32_t* cb = a.codes + (long)b * a.F * 7;
  float acc = a.dw_b[c];
#pragma unroll
  for (int k = 0; k < 7; ++k) {
    int tt = t + k - 3;
    if (tt < 0 || tt >= T) continue;
    int f = tt >> 2, j = tt & 3;
    const int32_t* fr = cb + f * 7;
    int c0 = clampc(fr[0]);
    int c1 = clampc(fr[(j >> 1) ? 4 : 1]);
    int c2 = clampc(fr[j == 0 ? 2 : j == 1 ? 3 : j == 2 ? 5 : 6]);
    float z = a.P0[(long)c0 * kLat + c] + a.P1[(long)c1 * kLat + c] + a.P2[(long)c2 * kLat + c];
    acc = fmaf(a.dw_w[c * 7 + k], z, acc);
  }
  a.z[i] = acc;
}

// Create-time: split a weight tensor into round-to-nearest TF32 hi and residual lo.
__global__ void k_split_w(const float* w, float* hi, float* lo, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = w[i], h = tf32r(v);
  hi[i] = h;
  lo[i] = tf32r(v - h);
}

// ---------------------------------------------------------------------------------------
// ResidualUnit front: a = snake2(dwconv_dil(snake1(x))), zero padded per batch item.
constexpr int DW_TT = 64, DW_CT = 32, DW_TY = 8, DW_HMAX = 27;
__global__ void __launch_bounds__(DW_CT* DW_TY)
    k_dw(const float* __restrict__ x, const float* __restrict__ a1, const float* __restrict__ dw_w,
         const float* __restrict__ dw_b, const float* __restrict__ a2, float* __restrict__ a_out,
         int T, int C, int dil) {
  __shared__ float s[DW_TT + 2 * DW_HMAX][DW_CT];
  griddep_wait();
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int c = blockIdx.y * DW_CT + tx, b = blockIdx.z, t0 = blockIdx.x * DW_TT;
  const int H = 3 * dil;
  const float al1 = a1[c], inv1 = snake_inv(al1);
  const float* xb = x + (long)b * T * C + c;
  for (int r = ty; r < DW_TT + 2 * H; r += DW_TY) {
    int t = t0 - H + r;
    s[r][tx] = (t >= 0 && t < T) ? snake(xb[(long)t * C], al1, inv1) : 0.f;
  }
  __syncthreads();
  float w[7];
#pragma unroll
  for (int k = 0; k < 7; ++k) w[k] = dw_w[c * 7 + k];
  const float bias = dw_b[c], al2 = a2[c], inv2 = snake_inv(al2);
#pragma unroll
  for (int j = 0; j < DW_TT / DW_TY; ++j) {
    int o = ty + j * DW_TY, t = t0 + o;
    if (t >= T) break;
    float acc = bias;
#pragma unroll
    for (int k = 0; k < 7; ++k) acc = fmaf(w[k], s[o + k * dil][tx], acc);
    a_out[((long)b * T + t) * C + c] = snake(acc, al2, inv2);
  }
}

// ---------------------------------------------------------------------------------------
// Tensor-core GEMM: out[M][N] = epi(Aop[M][K] @ W[N][K]^T), W pre-split into hi/lo.
enum { A_PLAIN = 0, A_CONVT = 1 };
enum { E_BIAS = 0, E_CONVT = 1, E_NOISE = 2, E_RU = 3 };

struct alignas(64) GemmP {
  // TMA descriptors: ta = A [rows][K] fp32 box {32, BM} (A_PLAIN only); tbh/tbl = weight
  // hi/lo planes [rows][K] box {32, BN}, 128B swizzle (rows = N, or s*N for ConvT).
  CUtensorMap ta, tbh, tbl;
  const float* A;  // activations, row-major K-contiguous (CONVT: X [B*Lin][Cin])
  const float* Whi;  // [N][K] (CONVT: [s][N][K]; phase from blockIdx.z)
  const float* Wlo;
  const float* bias;   // [N]
  const float* resid;  // E_RU / E_NOISE: [M][N]
  float* out;
  int M, N, K;
  int T;     // rows per batch item (E_NOISE, opad)
  int opad;  // store rows as [B][T+1][N] with a zero row closing each item (ConvT input)
  const float* alpha;  // CONVT: snake alpha [Cin]
  int Lin, s, Co;      // CONVT: input rows per item, stride, output channels (N = s*Co)
  const unsigned long long* seed_slot;  // E_NOISE
  int blk;
  int splits, kper;  // split-K factor (divides K/BK) and chunks per work item
  int gx, gy, items;  // m-tiles, n-tiles, total work items (= phases*gy*gx*splits)
  float* ws;          // split-K partials [tile][split][BM*BN]
  int* cnt;           // split-K arrival counters [tile]
};

constexpr int BM = 128, BK = 32, PRM = 1024;

__device__ __forceinline__ void tma2d(float* dst, const CUtensorMap* map, int c0, int c1, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n" ::
          "r"((uint32_t)__cvta_generic_to_shared(dst)),
      "l"(map), "r"(c0), "r"(c1), "r"((uint32_t)__cvta_generic_to_shared(bar))
      : "memory");
}
__device__ __forceinline__ void mbar_expect(uint64_t* b, int bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(
                   (uint32_t)__cvta_generic_to_shared(b)),
               "r"(bytes)
               : "memory");
}

// ---- Hopper wgmma (TF32, K-major, 128-byte swizzle; recipe of ../sm90_wgmma.cuh) --------
// Tile row = 32 fp32 = 128 B = one swizzle atom row; 16-byte chunk c of row r lives at
// float offset r*32 + ((c ^ (r & 7)) * 4). Descriptor LBO 16 B, SBO 1024 B, swizzle 128 B;
// a k8 substep advances the start address by 32 B. Tiles are 1024-byte aligned. wgmma reads
// fp32 bit patterns and ignores the low 13 bits, so operands are stored pre-rounded.
__device__ __forceinline__ uint64_t wg_desc(const float* p) {
  uint64_t a = (uint64_t)__cvta_generic_to_shared(p);
  return ((a & 0x3FFFFull) >> 4) | ((16ull >> 4) << 16) | ((1024ull >> 4) << 32) | (1ull << 62);
}
__device__ __forceinline__ int swz(int row, int c4) { return row * BK + ((c4 ^ (row & 7)) << 2); }
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
template <int N>
__device__ __forceinline__ void wg_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}
__device__ __forceinline__ void fence_async_smem() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}
__device__ __forceinline__ void wgmma_tf32_n128(float* d, uint64_t da, uint64_t db, int scale_d) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %66, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k8.f32.tf32.tf32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, p, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
      : "l"(da), "l"(db), "r"(scale_d));
}
__device__ __forceinline__ void wgmma_tf32_n64(float* d, uint64_t da, uint64_t db, int scale_d) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k8.f32.tf32.tf32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, p, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "l"(da), "l"(db), "r"(scale_d));
}
template <int BN>
__device__ __forceinline__ void wgmma_tf32(float* d, uint64_t da, uint64_t db, int scale_d) {
  if constexpr (BN == 128)
    wgmma_tf32_n128(d, da, db, scale_d);
  else
    wgmma_tf32_n64(d, da, db, scale_d);
}

// ---- warp-specialized persistent GEMM -------------------------------------------------
// 384 threads: warpgroup 0 = producer, warpgroups 1-2 = consumers (64 tile rows each).
// Work item = (phase, n-tile, m-tile, K-split); block b handles items b, b+G, ... as one
// continuous stream of 32-wide K chunks. Shared memory rings:
//   B ring  [3]: pre-split weight planes (hi, lo) by TMA (128B swizzle = the wgmma layout),
//                mbarrier fullB (tx bytes), freed by the consumers via emptyB.
//   raw ring[3]: raw fp32 activation chunks by TMA (fullR).
//   A planes[2]: producer transform raw -> (snake for ConvT) -> RN tf32 hi/lo planes,
//                fullA / emptyA mbarriers.
// Loads run 2 chunks ahead. Consumers: wait; 3 wgmma per k8 (lo*hi, hi*lo, hi*hi) into
// `part`; wait; release; acc += part (rounded fp32 add per chunk, so the tensor core's
// truncating accumulation does not bias long K); epilogue after an item's last chunk
// (split-K: deterministic last-arriver fixup) while the producer streams the next item.
constexpr int NTH = 384;
__device__ __forceinline__ uint32_t su32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void mbar_init(uint64_t* b, int cnt) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(su32(b)), "r"(cnt));
}
__device__ __forceinline__ void mbar_arrive(uint64_t* b) {
  asm volatile("{\n.reg .b64 st;\nmbarrier.arrive.shared::cta.b64 st, [%0];\n}\n" ::"r"(su32(b))
               : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, int parity) {
  asm volatile(
      "{\n.reg .pred p;\nWAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra WAIT_%=;\n}\n" ::"r"(su32(b)),
      "r"(parity)
      : "memory");
}
__device__ __forceinline__ void bar_consumers() {  // named barrier over the 256 consumers
  asm volatile("bar.sync 1, 256;\n" ::: "memory");
}

template <int BN, bool SPLIT>
struct Smem {
  static constexpr int NPL = SPLIT ? 2 : 1;
  static constexpr int NB = 3, NR = 3, NA = 2;
  static constexpr int APL = BM * BK, BPL = BN * BK;
  static constexpr int B0 = 0, A0 = B0 + NB * NPL * BPL, R0 = A0 + NA * NPL * APL,
                       P0 = R0 + NR * APL, BAR0 = P0 + PRM, END = BAR0 + 2 * (2 * NB + 2 * NA + NR) + 4;
  static constexpr int bytes = END * 4 + 1024;
};

template <int AM, int EM, int BN, bool SPLIT>
__global__ void __launch_bounds__(NTH, 1) k_gemm(const __grid_constant__ GemmP p) {
  using S = Smem<BN, SPLIT>;
  constexpr int NB = S::NB, NR = S::NR, NACC = BN / 2;
  constexpr int A_V4 = BM * BK / 4 / 128;          // 8 raw float4 per producer thread
  constexpr int B_V4 = S::NPL * BN * BK / 4 / 128;  // 16/8 (split) or 8/4
  extern __shared__ float sm_raw[];
  float* sm = (float*)(((uintptr_t)sm_raw + 1023) & ~(uintptr_t)1023);
  uint64_t* fullB = reinterpret_cast<uint64_t*>(sm + S::BAR0);
  uint64_t* emptyB = fullB + NB;
  uint64_t* fullA = emptyB + NB;
  uint64_t* emptyA = fullA + 2;
  uint64_t* fullR = emptyA + 2;  // raw activation chunk landed
  int* flag = reinterpret_cast<int*>(fullR + NR);

  const int tid = threadIdx.x, wg = tid >> 7;
  const int K = p.K, M = p.M, N = p.N, kper = p.kper;
  const int gx = p.gx, gy = p.gy, G = gridDim.x;
  const int Cin = K / 2;
  const int my_items = (p.items - (int)blockIdx.x + G - 1) / G;
  const int total = my_items * kper;

  if (tid == 0) {
    for (int s = 0; s < NB; ++s) {
      mbar_init(fullB + s, 1);  // TMA: one expect_tx arrival + transaction bytes
      mbar_init(emptyB + s, 8);  // one arrival per consumer warp
    }
    for (int s = 0; s < NR; ++s) mbar_init(fullR + s, 1);
    for (int s = 0; s < 2; ++s) {
      mbar_init(fullA + s, 4);  // one arrival per producer warp
      mbar_init(emptyA + s, 8);
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  if (AM == A_CONVT)
    for (int i = tid; i < Cin; i += NTH) sm[S::P0 + i] = p.alpha[i];
  __syncthreads();
  // PDL: everything above overlapped the previous kernel; activations are read below.
  griddep_wait();
  griddep_launch_dependents();

  struct Item {
    int m0, n0, phase, kz, tile;
  };
  auto decode = [&](int q) {
    int it = (int)blockIdx.x + (q / kper) * G;
    Item r;
    r.kz = it % p.splits;
    int t = it / p.splits;
    r.tile = t;
    r.m0 = (t % gx) * BM;
    t /= gx;
    r.n0 = (t % gy) * BN;
    r.phase = t / gy;
    return r;
  };

  if (wg == 0) {
    // ============================== producer ==============================
    const int pt = tid, k4 = pt & 7;
    // Issue chunk q (thread 0): weight planes (B ring) and the raw activation chunk (raw
    // ring) by TMA. ConvT: the input is stored [B][Lin+1][Cin] with a zero row closing each
    // batch item, so row m = (b, q) of K-half 0 is flat row m and of K-half 1 is flat row
    // m-1 (row -1 is TMA zero fill): both are plain 2D boxes.
    auto issue = [&](int q) {
      if (q >= total || pt != 0) return;
      const Item it = decode(q);
      const int k0 = (it.kz * kper + q % kper) * BK;
      const int sb = q % NB, sr = q % NR;
      if (q >= NB) mbar_wait(emptyB + sb, ((q / NB) - 1) & 1);
      float* bdst = sm + S::B0 + sb * S::NPL * S::BPL;
      const int brow = (AM == A_CONVT ? it.phase * N : 0) + it.n0;
      mbar_expect(fullB + sb, S::NPL * S::BPL * 4);
      tma2d(bdst, &p.tbh, k0, brow, fullB + sb);
      if (SPLIT) tma2d(bdst + S::BPL, &p.tbl, k0, brow, fullB + sb);
      float* raw = sm + S::R0 + sr * S::APL;
      mbar_expect(fullR + sr, S::APL * 4);
      if (AM == A_CONVT) {
        const int hi = k0 >= Cin;
        tma2d(raw, &p.ta, k0 - hi * Cin, it.m0 - hi, fullR + sr);
      } else {
        tma2d(raw, &p.ta, k0, it.m0, fullR + sr);
      }
    };
    issue(0);
    issue(1);
    for (int q = 0; q < total; ++q) {
      mbar_wait(fullR + q % NR, (q / NR) & 1);
      const int sa = q & 1;
      if (q >= 2) mbar_wait(emptyA + sa, ((q >> 1) - 1) & 1);
      const float* raw = sm + S::R0 + (q % NR) * S::APL;
      float* pl = sm + S::A0 + sa * S::NPL * S::APL;
      float4 al = make_float4(0.f, 0.f, 0.f, 0.f), inv = al;
      if (AM == A_CONVT) {
        const Item it = decode(q);
        const int k0 = (it.kz * kper + q % kper) * BK;
        int kk = (k0 >= Cin ? k0 - Cin : k0) + k4 * 4;
        al = *reinterpret_cast<const float4*>(sm + S::P0 + kk);
        inv = make_float4(snake_inv(al.x), snake_inv(al.y), snake_inv(al.z), snake_inv(al.w));
      }
#pragma unroll
      for (int j = 0; j < A_V4; ++j) {
        int r = pt / 8 + j * 16;
        float4 x = *reinterpret_cast<const float4*>(raw + r * BK + k4 * 4);
        if (AM == A_CONVT) {  // zero rows stay zero: snake(0) = 0
          x.x = snake(x.x, al.x, inv.x);
          x.y = snake(x.y, al.y, inv.y);
          x.z = snake(x.z, al.z, inv.z);
          x.w = snake(x.w, al.w, inv.w);
        }
        float4 h = make_float4(tf32r(x.x), tf32r(x.y), tf32r(x.z), tf32r(x.w));
        *reinterpret_cast<float4*>(pl + swz(r, k4)) = h;
        if (SPLIT) {
          float4 l = make_float4(tf32r(x.x - h.x), tf32r(x.y - h.y), tf32r(x.z - h.z), tf32r(x.w - h.w));
          *reinterpret_cast<float4*>(pl + S::APL + swz(r, k4)) = l;
        }
      }
      fence_async_smem();
      __syncwarp();
      if ((pt & 31) == 0) mbar_arrive(fullA + sa);
      // Refill chunk q+2 (thread 0): its raw slot is chunk q-1's (every producer is past it:
      // named barrier) and its B slot is chunk q-1's (thread 0 waits for the consumers'
      // release; the other producers go on transforming meanwhile).
      asm volatile("bar.sync 2, 128;\n" ::: "memory");
      issue(q + 2);
    }
    return;
  }

  // ============================== consumers ==============================
  const int ct = tid - 128, cw = ct >> 7, warp = (ct >> 5) & 3, lane = ct & 31;
  unsigned long long seed = 0;
  if (EM == E_NOISE) seed = *p.seed_slot;
  float acc[NACC], part[NACC];
#pragma unroll
  for (int i = 0; i < NACC; ++i) acc[i] = part[i] = 0.f;

  auto epilogue = [&](const Item& it) {
    if (p.splits > 1) {
      // Publish this split's partial ([split][j][thread] float4, coalesced); the last
      // arriving split sums all partials in split order (deterministic) and stores.
      float* base = p.ws + (long)it.tile * p.splits * (BM * BN);
      float4* mine = reinterpret_cast<float4*>(base + (long)it.kz * (BM * BN)) + ct;
#pragma unroll
      for (int j = 0; j < NACC / 4; ++j)
        __stcg(mine + j * 256, make_float4(acc[4 * j], acc[4 * j + 1], acc[4 * j + 2], acc[4 * j + 3]));
      __threadfence();
      bar_consumers();
      if (ct == 0) {
        int old = atomicAdd(p.cnt + it.tile, 1);
        *flag = old == p.splits - 1;
        if (*flag) p.cnt[it.tile] = 0;  // every split arrived: reset for the next launch
      }
      bar_consumers();
      const bool last = *flag;
      bar_consumers();  // flag is rewritten by the next item's epilogue
      if (!last) return;
      __threadfence();
      float4 sum[NACC / 4];
#pragma unroll
      for (int j = 0; j < NACC / 4; ++j) sum[j] = make_float4(0.f, 0.f, 0.f, 0.f);
      for (int z = 0; z < p.splits; ++z) {
        float4 v[NACC / 4];
        const float4* src = reinterpret_cast<const float4*>(base + (long)z * (BM * BN)) + ct;
#pragma unroll
        for (int j = 0; j < NACC / 4; ++j) v[j] = __ldcg(src + j * 256);
        if (z == it.kz) {
#pragma unroll
          for (int j = 0; j < NACC / 4; ++j)
            v[j] = make_float4(acc[4 * j], acc[4 * j + 1], acc[4 * j + 2], acc[4 * j + 3]);
        }
#pragma unroll
        for (int j = 0; j < NACC / 4; ++j) {
          sum[j].x += v[j].x;
          sum[j].y += v[j].y;
          sum[j].z += v[j].z;
          sum[j].w += v[j].w;
        }
      }
#pragma unroll
      for (int j = 0; j < NACC / 4; ++j) {
        acc[4 * j] = sum[j].x;
        acc[4 * j + 1] = sum[j].y;
        acc[4 * j + 2] = sum[j].z;
        acc[4 * j + 3] = sum[j].w;
      }
    }
    // m64nBN accumulator, warp w of the warpgroup: reg 4g+2h+l ->
    // row 16w + lane/4 + 8h, col 8g + 2(lane%4) + l.
    const int Lout = p.Lin * p.s;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      int m = it.m0 + cw * 64 + warp * 16 + (lane >> 2) + 8 * h;
      if (m >= M) continue;
      if (EM == E_CONVT) {
        // All output phases in one GEMM: column n = r*Co + co is output row qq*s + r - s/2.
        const int b = m / (p.Lin + 1), qq = m - b * (p.Lin + 1), Co = p.Co;
        float2 bb[BN / 8];
#pragma unroll
        for (int g = 0; g < BN / 8; ++g) {
          int n = it.n0 + g * 8 + (lane & 3) * 2;
          bb[g] = __ldg(reinterpret_cast<const float2*>(p.bias + n % Co));
        }
#pragma unroll
        for (int g = 0; g < BN / 8; ++g) {
          int n = it.n0 + g * 8 + (lane & 3) * 2, r = n / Co, co = n - r * Co;
          int to = qq * p.s + r - p.s / 2;
          if (to < 0 || to >= Lout) continue;
          float2 v = make_float2(acc[4 * g + 2 * h] + bb[g].x, acc[4 * g + 2 * h + 1] + bb[g].y);
          *reinterpret_cast<float2*>(p.out + ((long)b * Lout + to) * Co + co) = v;
        }
        continue;
      }
      long orow = m;
      bool zrow = false;  // opad: also clear the separator row after the batch item's last row
      if (p.opad) {
        int b = m / p.T, t = m - b * p.T;
        orow = (long)b * (p.T + 1) + t;
        zrow = t == p.T - 1;
      }
      float nz = 0.f;
      if (EM == E_NOISE) nz = noise_at(seed, p.blk, m / p.T, m % p.T);
      // Batch every load of this row before any store: `out` may alias nothing we read,
      // but the compiler cannot know, and interleaving would serialize one L2 round trip
      // per store.
      float2 x[BN / 8];
      const float* rrow = p.resid + (long)m * N + it.n0 + (lane & 3) * 2;
      const float* brow = p.bias + it.n0 + (lane & 3) * 2;
#pragma unroll
      for (int g = 0; g < BN / 8; ++g) {
        if (EM == E_NOISE || EM == E_RU)
          x[g] = __ldg(reinterpret_cast<const float2*>(rrow + g * 8));
        else
          x[g] = make_float2(0.f, 0.f);
        if (EM != E_NOISE) {
          float2 bb = __ldg(reinterpret_cast<const float2*>(brow + g * 8));
          x[g].x += bb.x;
          x[g].y += bb.y;
        }
      }
      float* orow_p = p.out + orow * N + it.n0 + (lane & 3) * 2;
#pragma unroll
      for (int g = 0; g < BN / 8; ++g) {
        float2 v = make_float2(acc[4 * g + 2 * h], acc[4 * g + 2 * h + 1]);
        if (EM == E_NOISE) {
          v.x = fmaf(nz, v.x, x[g].x);
          v.y = fmaf(nz, v.y, x[g].y);
        } else {
          v.x += x[g].x;
          v.y += x[g].y;
        }
        *reinterpret_cast<float2*>(orow_p + g * 8) = v;
        if (zrow) *reinterpret_cast<float2*>(orow_p + N + g * 8) = make_float2(0.f, 0.f);
      }
    }
  };

  for (int q = 0; q < total; ++q) {
    const int sa = q & 1, sb = q % NB;
    mbar_wait(fullA + sa, (q >> 1) & 1);
    mbar_wait(fullB + sb, (q / NB) & 1);
    fence_async_smem();
    const float* Ah = sm + S::A0 + sa * S::NPL * S::APL + cw * 64 * BK;
    const float* Bh = sm + S::B0 + sb * S::NPL * S::BPL;
    wg_fence();
#pragma unroll
    for (int ks = 0; ks < BK / 8; ++ks) {
      const uint64_t dah = wg_desc(Ah + ks * 8), dbh = wg_desc(Bh + ks * 8);
      if (SPLIT) {
        const uint64_t dal = wg_desc(Ah + S::APL + ks * 8), dbl = wg_desc(Bh + S::BPL + ks * 8);
        wgmma_tf32<BN>(part, dal, dbh, ks > 0);
        wgmma_tf32<BN>(part, dah, dbl, 1);
        wgmma_tf32<BN>(part, dah, dbh, 1);
      } else {
        wgmma_tf32<BN>(part, dah, dbh, ks > 0);
      }
    }
    wg_commit();
    wg_wait<0>();
    // One arrival per warp: 256 threads hammering one mbarrier word serialize badly.
    if (lane == 0) {
      mbar_arrive(emptyA + sa);
      mbar_arrive(emptyB + sb);
    }
    // Tensor-core accumulation truncates; fold each 32-wide K chunk into the fp32
    // accumulator with a correctly rounded add so the bias does not grow with K.
    const bool first = q % kper == 0;
#pragma unroll
    for (int j = 0; j < NACC; ++j) acc[j] = first ? part[j] : acc[j] + part[j];
    if (q % kper == kper - 1) epilogue(decode(q));
  }
}

// ---------------------------------------------------------------------------------------
// Final: Snake -> conv 64->1 k7 pad3 -> tanh.
struct FinalArgs {
  const float* x;  // [B][T][64]
  int T;
  const float* alpha;
  const float* w;  // [64][7]
  const float* b;
  float* pcm;  // [B][T]
};
constexpr int FIN_TT = 128, FIN_C = 64;
__global__ void __launch_bounds__(FIN_TT) k_final(FinalArgs a) {
  __shared__ float s[FIN_TT + 6][FIN_C + 1];
  __shared__ float ws[FIN_C * 7];
  griddep_wait();
  const int b = blockIdx.y, t0 = blockIdx.x * FIN_TT, tid = threadIdx.x;
  for (int i = tid; i < FIN_C * 7; i += FIN_TT) ws[i] = a.w[i];
  const float* xb = a.x + (long)b * a.T * FIN_C;
  const int c = tid % FIN_C;
  const float al = a.alpha[c], inv = snake_inv(al);
  for (int i = tid; i < (FIN_TT + 6) * FIN_C; i += FIN_TT) {
    int r = i / FIN_C, t = t0 - 3 + r;
    s[r][c] = (t >= 0 && t < a.T) ? snake(xb[(long)t * FIN_C + c], al, inv) : 0.f;
  }
  __syncthreads();
  int t = t0 + tid;
  if (t >= a.T) return;
  float acc = a.b[0];
#pragma unroll 8
  for (int cc = 0; cc < FIN_C; ++cc) {
#pragma unroll
    for (int k = 0; k < 7; ++k) acc = fmaf(ws[cc * 7 + k], s[tid + k][cc], acc);
  }
  a.pcm[(long)b * a.T + t] = tanhf(acc);
}

// ---------------------------------------------------------------------------------------
struct Wsp {  // a GEMM weight, pre-split
  const float* hi = nullptr;
  const float* lo = nullptr;
};
struct RUW {
  const float *a1, *dw_w, *dw_b, *a2, *pw_b;
  Wsp pw;
};
struct BlkW {
  const float *snake, *up_b;
  Wsp up, noise;
  RUW ru[3];
};

struct GraphEntry {
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  cudaGraphNode_t quant_node = nullptr, final_node = nullptr;
  const float* final_in = nullptr;
};

constexpr long kWsFloats = 16L << 20;  // split-K partials (64 MiB)
struct Snac;
long kBufFloats(const Snac* s);
constexpr int kMaxTiles = 8192;        // split-K tile counters

struct Snac {
  int device = 0, max_batch = 0, max_frames = 0;
  bool use_graph = true, split = true;
  int dbg_stop = -1;  // test hook: PLOW_SNAC_DEBUG_STOP=stage ends enqueue early
  const float* dbg_out = nullptr;
  float* wbuf = nullptr;
  float* wsplit = nullptr;
  std::map<std::string, std::pair<const float*, uint64_t>> w;
  const float *P[3], *pre_dw_w, *pre_dw_b, *pre_pw_b, *out_snake, *out_w, *out_b;
  Wsp pre_pw;
  BlkW blk[4];
  float *xA = nullptr, *xB = nullptr, *xC = nullptr;
  long nf = 0;  // frame capacity (max_batch * max_frames)
  unsigned long long* seed_slot = nullptr;
  float* ws = nullptr;
  int* cnt = nullptr;
  cudaStream_t cap_stream = nullptr;
  // plow_snac_decode_host: own stream, pinned staging, device codes/pcm buffers.
  cudaStream_t host_stream = nullptr;
  int32_t* h_codes = nullptr;
  float* h_pcm = nullptr;
  int32_t* d_codes = nullptr;
  float* d_pcm = nullptr;
  std::map<std::tuple<int, int, int>, GraphEntry> graphs;
  PFN_cuTensorMapEncodeTiled_v12000 encode = nullptr;
  std::map<std::tuple<const void*, long, long, int, int>, CUtensorMap> maps;
};

// Activation buffer size: 128K floats per frame (largest stage 1024x128 / 2048x64) plus
// the zero separator rows of padded ConvT inputs.
long kBufFloats(const Snac* s) { return s->nf * 128 * 1024 + (long)s->max_batch * 1024; }

// 2D fp32 tensor map [rows][cols] with a {32, box_rows} box (cached; host-only encode).
bool tmap(Snac* s, CUtensorMap* out, const float* base, long cols, long rows, int box_rows, bool swz) {
  auto key = std::make_tuple((const void*)base, cols, rows, box_rows, (int)swz);
  auto it = s->maps.find(key);
  if (it != s->maps.end()) {
    *out = it->second;
    return true;
  }
  if (!s->encode) {
    cudaDriverEntryPointQueryResult qr;
    if (cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", (void**)&s->encode, cudaEnableDefault, &qr) !=
            cudaSuccess ||
        qr != cudaDriverEntryPointSuccess)
      return false;
  }
  cuuint64_t dims[2] = {(cuuint64_t)cols, (cuuint64_t)rows};
  cuuint64_t strides[1] = {(cuuint64_t)cols * 4};
  cuuint32_t box[2] = {(cuuint32_t)BK, (cuuint32_t)box_rows};
  cuuint32_t es[2] = {1, 1};
  CUresult r = s->encode(out, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, (void*)base, dims, strides, box, es,
                         CU_TENSOR_MAP_INTERLEAVE_NONE,
                         swz ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE,
                         CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) return false;
  s->maps[key] = *out;
  return true;
}

#define CK(x)                                                                              \
  do {                                                                                     \
    cudaError_t e_ = (x);                                                                  \
    if (e_ != cudaSuccess) {                                                               \
      fprintf(stderr, "plow_snac: %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
      return -2;                                                                           \
    }                                                                                      \
  } while (0)

inline unsigned blocks(long n, int t) { return (unsigned)((n + t - 1) / t); }

// Launch with programmatic stream serialization (PDL) unless PLOW_SNAC_PDL=0.
bool g_pdl = true;
template <typename... KArgs, typename... Args>
cudaError_t launch(void (*k)(KArgs...), dim3 g, dim3 b, size_t smem, cudaStream_t st, Args... args) {
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = g;
  cfg.blockDim = b;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = st;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = g_pdl ? 1 : 0;
  return cudaLaunchKernelEx(&cfg, k, args...);
}

template <int AM, int EM, int BN, bool SPLIT>
void launch_t(const GemmP& p, cudaStream_t st) {
  launch(k_gemm<AM, EM, BN, SPLIT>, dim3(std::min(p.items, kSMs)), dim3(NTH), Smem<BN, SPLIT>::bytes, st, p);
}
// Picks BN and a split-K factor (a divisor of K/BK, >= 4 chunks per split) so that small-M
// GEMMs still spread over the GPU; the persistent grid covers the work items.
template <int AM, int EM>
int gemm(Snac* s, GemmP p, int phases, cudaStream_t st) {
  const int bn = p.N % 128 == 0 ? 128 : 64;
  if (p.N % bn || p.K % BK || (AM == A_CONVT && ((p.K / 2) % BK || p.K / 2 > PRM))) return -1;
  p.gx = (int)blocks(p.M, BM);
  p.gy = p.N / bn;
  const int tiles = p.gx * p.gy * phases, nk = p.K / BK;
  int splits = 1;
  if (tiles < kSMs && tiles <= kMaxTiles) {
    const int want = std::min((kSMs + tiles - 1) / tiles, 16);
    for (int d = 2; d <= want && d * 4 <= nk; ++d)
      if (nk % d == 0 && (long)tiles * d * BM * bn <= kWsFloats) splits = d;
  }
  p.splits = splits;
  p.kper = nk / splits;
  p.items = tiles * splits;
  p.ws = s->ws;
  p.cnt = s->cnt;
  // A rows: whole scratch buffer (rows >= M are read but never stored); weights: N*phases.
  {
    const long cols = AM == A_CONVT ? p.K / 2 : p.K;
    if (!tmap(s, &p.ta, p.A, cols, kBufFloats(s) / cols, BM, false)) return -2;
  }
  if (!tmap(s, &p.tbh, p.Whi, p.K, (long)p.N * phases, bn, true)) return -2;
  if (!tmap(s, &p.tbl, p.Wlo, p.K, (long)p.N * phases, bn, true)) return -2;
  if (bn == 128) {
    if (s->split)
      launch_t<AM, EM, 128, true>(p, st);
    else
      launch_t<AM, EM, 128, false>(p, st);
  } else {
    if (s->split)
      launch_t<AM, EM, 64, true>(p, st);
    else
      launch_t<AM, EM, 64, false>(p, st);
  }
  CK(cudaGetLastError());
  return 0;
}

template <int AM, int EM, int BN, bool SPLIT>
int set_attr_t() {
  CK(cudaFuncSetAttribute(k_gemm<AM, EM, BN, SPLIT>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                          Smem<BN, SPLIT>::bytes));
  return 0;
}
template <int AM, int EM>
int set_attr() {
  int r = set_attr_t<AM, EM, 128, true>();
  r = r ? r : set_attr_t<AM, EM, 128, false>();
  r = r ? r : set_attr_t<AM, EM, 64, true>();
  r = r ? r : set_attr_t<AM, EM, 64, false>();
  return r;
}

// While capturing, returns the node just captured on `st` (the stream's sole dependency).
int last_node(cudaStream_t st, cudaGraphNode_t* out) {
  cudaStreamCaptureStatus cs;
  const cudaGraphNode_t* deps = nullptr;
  size_t nd = 0;
  CK(cudaStreamGetCaptureInfo(st, &cs, nullptr, nullptr, &deps, &nd));
  if (cs != cudaStreamCaptureStatusActive || nd != 1) return -2;
  *out = deps[0];
  return 0;
}

int enqueue(Snac* s, const int32_t* codes, int B, int F, float* pcm, unsigned long long seed,
            cudaStream_t st, GraphEntry* ge = nullptr) {
  const bool noise = seed != 0;
  int T = 4 * F;
  long N = (long)B * T;
  // Buffers: cur = block activation, oth = next activation, tmp = RU dw output.
  float* cur = s->xB;
  float* oth = s->xA;
  float* tmp = s->xC;
  QuantArgs qa{codes, B, F, s->P[0], s->P[1], s->P[2], s->pre_dw_w, s->pre_dw_b, tmp, seed,
               s->seed_slot};
  k_quant<<<blocks(N * kLat, 256), 256, 0, st>>>(qa);
  CK(cudaGetLastError());
  if (ge)
    if (int r = last_node(st, &ge->quant_node)) return r;
  GemmP p{};
  p.A = tmp;
  p.Whi = s->pre_pw.hi;
  p.Wlo = s->pre_pw.lo;
  p.bias = s->pre_pw_b;
  p.out = cur;
  p.M = (int)N;
  p.N = kD0;
  p.K = kLat;
  p.T = T;
  p.opad = 1;  // feeds ConvT 0
  if (int r = gemm<A_PLAIN, E_BIAS>(s, p, 1, st)) return r;
  int stage = 0;
  if (s->dbg_stop == stage) return s->dbg_out = cur, 0;
  int C = kD0;
  for (int i = 0; i < 4; ++i) {
    const BlkW& bw = s->blk[i];
    const int sd = kStride[i], Co = C / 2;
    GemmP u{};
    u.A = cur;
    u.Whi = bw.up.hi;
    u.Wlo = bw.up.lo;
    u.bias = bw.up_b;
    u.out = oth;
    u.M = B * (T + 1);
    u.N = sd * Co;  // all phases in one GEMM (weights are [s][Co][2C] = [N][K])
    u.Co = Co;
    u.K = 2 * C;
    u.alpha = bw.snake;
    u.Lin = T;
    u.s = sd;
    if (int r = gemm<A_CONVT, E_CONVT>(s, u, 1, st)) return r;
    std::swap(cur, oth);
    if (s->dbg_stop == ++stage) return s->dbg_out = cur, 0;
    T *= sd;
    N = (long)B * T;
    C = Co;
    if (noise) {
      GemmP q{};
      q.A = cur;
      q.Whi = bw.noise.hi;
      q.Wlo = bw.noise.lo;
      q.resid = cur;
      q.out = oth;
      q.M = (int)N;
      q.N = C;
      q.K = C;
      q.T = T;
      q.seed_slot = s->seed_slot;
      q.blk = i;
      if (int r = gemm<A_PLAIN, E_NOISE>(s, q, 1, st)) return r;
      std::swap(cur, oth);
    }
    for (int j = 0; j < 3; ++j) {
      const RUW& rw = bw.ru[j];
      dim3 g(blocks(T, DW_TT), C / DW_CT, B), tb(DW_CT, DW_TY);
      CK(launch(k_dw, g, tb, 0, st, (const float*)cur, rw.a1, rw.dw_w, rw.dw_b, rw.a2, tmp, T, C, kDil[j]));
      GemmP r{};
      r.A = tmp;
      r.Whi = rw.pw.hi;
      r.Wlo = rw.pw.lo;
      r.bias = rw.pw_b;
      r.resid = cur;
      r.out = oth;
      r.M = (int)N;
      r.N = C;
      r.K = C;
      r.T = T;
      r.opad = j == 2 && i < 3;  // feeds the next block's ConvT
      if (int rc = gemm<A_PLAIN, E_RU>(s, r, 1, st)) return rc;
      std::swap(cur, oth);
      if (s->dbg_stop == ++stage) return s->dbg_out = cur, 0;
    }
  }
  FinalArgs fa{cur, T, s->out_snake, s->out_w, s->out_b, pcm};
  CK(launch(k_final, dim3(blocks(T, FIN_TT), B), dim3(FIN_TT), 0, st, fa));
  if (ge) {
    ge->final_in = cur;
    if (int r = last_node(st, &ge->final_node)) return r;
  }
  return 0;
}

int build_graph(Snac* s, int B, int F, bool noise, GraphEntry* ge) {
  cudaGraph_t g = nullptr;
  CK(cudaStreamBeginCapture(s->cap_stream, cudaStreamCaptureModeThreadLocal));
  int r = enqueue(s, nullptr, B, F, nullptr, noise ? 1ULL : 0ULL, s->cap_stream, ge);
  cudaError_t e = cudaStreamEndCapture(s->cap_stream, &g);
  if (r || e != cudaSuccess) {
    if (g) cudaGraphDestroy(g);
    if (r) return r;
    CK(e);
  }
  cudaGraphExec_t ex = nullptr;
  if (cudaGraphInstantiate(&ex, g, 0) != cudaSuccess) {
    cudaGraphDestroy(g);
    return -2;
  }
  ge->graph = g;  // kept alive: exec node updates name its nodes
  ge->exec = ex;
  return 0;
}

int load_weights(Snac* s, const char* path) {
  FILE* f = fopen(path, "rb");
  if (!f) return -4;
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (sz < 16) {
    fclose(f);
    return -5;
  }
  std::vector<char> buf(sz);
  bool ok = fread(buf.data(), 1, sz, f) == (size_t)sz;
  fclose(f);
  if (!ok) return -4;
  if (memcmp(buf.data(), "SNAC24K1", 8)) return -5;
  uint32_t ver, cnt;
  memcpy(&ver, &buf[8], 4);
  memcpy(&cnt, &buf[12], 4);
  if (ver != 2) return -5;
  long p = 16;
  struct Rec {
    std::string name;
    uint64_t off, nb;
  };
  std::vector<Rec> recs;
  for (uint32_t i = 0; i < cnt; ++i) {
    uint32_t nl, nd;
    if (p + 4 > sz) return -5;
    memcpy(&nl, &buf[p], 4);
    p += 4;
    if (p + nl + 4 > sz) return -5;
    std::string name(&buf[p], nl);
    p += nl;
    memcpy(&nd, &buf[p], 4);
    p += 4 + 4L * nd;
    Rec r{name, 0, 0};
    if (p + 16 > sz) return -5;
    memcpy(&r.off, &buf[p], 8);
    memcpy(&r.nb, &buf[p + 8], 8);
    p += 16;
    if (r.off + r.nb > (uint64_t)sz || r.off % 16) return -5;
    recs.push_back(r);
  }
  CK(cudaMalloc(&s->wbuf, sz));
  CK(cudaMemcpy(s->wbuf, buf.data(), sz, cudaMemcpyHostToDevice));
  for (auto& r : recs) s->w[r.name] = {(const float*)((char*)s->wbuf + r.off), r.nb};
  return 0;
}

}  // namespace

extern "C" int plow_snac_destroy(void* hp);

extern "C" int plow_snac_create(int device, const char* weights_path, int max_batch,
                                int max_frames, void** out) {
  if (!out || !weights_path || max_batch < 1 || max_frames < 1) return -1;
  *out = nullptr;
  Snac* s = new Snac();
  s->device = device;
  s->max_batch = max_batch;
  s->max_frames = max_frames;
  const char* g = getenv("PLOW_SNAC_GRAPH");
  s->use_graph = !(g && g[0] == '0');
  if (const char* d = getenv("PLOW_SNAC_DEBUG_STOP")) {
    s->dbg_stop = atoi(d);
    s->use_graph = false;
  }
  const char* pr = getenv("PLOW_SNAC_PREC");
  s->split = !(pr && strcmp(pr, "tf32") == 0);
  if (const char* e = getenv("PLOW_SNAC_PDL")) g_pdl = e[0] != '0';
  auto fail = [&](int rc) {
    plow_snac_destroy(s);
    return rc;
  };
  if (cudaSetDevice(device) != cudaSuccess) return fail(-2);
  if (int r = load_weights(s, weights_path)) return fail(r);
  auto W = [&](const std::string& n, const float** dst, long numel) {
    auto it = s->w.find(n);
    if (it == s->w.end() || it->second.second != (uint64_t)numel * 4) {
      fprintf(stderr, "plow_snac: missing or mis-sized tensor %s\n", n.c_str());
      return false;
    }
    *dst = it->second.first;
    return true;
  };
  // GEMM weights to pre-split: (source, element count, destination).
  std::vector<std::tuple<const float*, long, Wsp*>> gw;
  const float* tmp = nullptr;
  bool ok = W("q.P0", &s->P[0], 4096L * kLat) && W("q.P1", &s->P[1], 4096L * kLat) &&
            W("q.P2", &s->P[2], 4096L * kLat) && W("pre.dw.w", &s->pre_dw_w, kLat * 7) &&
            W("pre.dw.b", &s->pre_dw_b, kLat) && W("pre.pw.w", &tmp, (long)kD0 * kLat) &&
            W("pre.pw.b", &s->pre_pw_b, kD0) && W("out.snake", &s->out_snake, 64) &&
            W("out.w", &s->out_w, 64 * 7) && W("out.b", &s->out_b, 1);
  if (ok) gw.emplace_back(tmp, (long)kD0 * kLat, &s->pre_pw);
  int C = kD0;
  for (int i = 0; i < 4 && ok; ++i) {
    std::string p = "blk" + std::to_string(i);
    BlkW& b = s->blk[i];
    int Co = C / 2;
    const float *up = nullptr, *nz = nullptr;
    long nup = (long)kStride[i] * Co * 2 * C;
    ok = W(p + ".snake", &b.snake, C) && W(p + ".up.w", &up, nup) && W(p + ".up.b", &b.up_b, Co) &&
         W(p + ".noise.w", &nz, (long)Co * Co);
    if (ok) {
      gw.emplace_back(up, nup, &b.up);
      gw.emplace_back(nz, (long)Co * Co, &b.noise);
    }
    for (int j = 0; j < 3 && ok; ++j) {
      std::string q = p + ".ru" + std::to_string(j);
      RUW& r = b.ru[j];
      const float* pw = nullptr;
      ok = W(q + ".a1", &r.a1, Co) && W(q + ".dw.w", &r.dw_w, Co * 7) && W(q + ".dw.b", &r.dw_b, Co) &&
           W(q + ".a2", &r.a2, Co) && W(q + ".pw.w", &pw, (long)Co * Co) &&
           W(q + ".pw.b", &r.pw_b, Co);
      if (ok) gw.emplace_back(pw, (long)Co * Co, &r.pw);
    }
    C = Co;
  }
  if (!ok) return fail(-5);
  long total = 0;
  for (auto& t : gw) total += std::get<1>(t);
  if (cudaMalloc(&s->wsplit, total * 2 * 4) != cudaSuccess) return fail(-2);
  long off = 0;
  for (auto& t : gw) {
    long n = std::get<1>(t);
    float* hi = s->wsplit + off;
    float* lo = s->wsplit + total + off;
    k_split_w<<<blocks(n, 256), 256>>>(std::get<0>(t), hi, lo, n);
    std::get<2>(t)->hi = hi;
    std::get<2>(t)->lo = lo;
    off += n;
  }
  if (cudaGetLastError() != cudaSuccess) return fail(-2);
  // Scratch: three activation buffers of 128K floats per frame (largest stage: 1024x128 or
  // 2048x64 per frame; also holds the 16x768 latent and 4x1024 pre-block activation).
  s->nf = (long)max_batch * max_frames;
  const long bufb = kBufFloats(s) * 4;
  if (cudaMalloc(&s->xA, bufb) != cudaSuccess || cudaMalloc(&s->xB, bufb) != cudaSuccess ||
      cudaMalloc(&s->xC, bufb) != cudaSuccess ||
      cudaMalloc(&s->seed_slot, 8) != cudaSuccess || cudaMalloc(&s->ws, kWsFloats * 4) != cudaSuccess ||
      cudaMalloc(&s->cnt, kMaxTiles * 4) != cudaSuccess ||
      cudaMemset(s->cnt, 0, kMaxTiles * 4) != cudaSuccess)
    return fail(-2);
  if (cudaStreamCreateWithFlags(&s->cap_stream, cudaStreamNonBlocking) != cudaSuccess)
    return fail(-2);
  int r = set_attr<A_PLAIN, E_BIAS>();
  r = r ? r : set_attr<A_PLAIN, E_NOISE>();
  r = r ? r : set_attr<A_PLAIN, E_RU>();
  r = r ? r : set_attr<A_CONVT, E_CONVT>();
  if (r) return fail(r);
  {
    const long ncodes = s->nf * 7, npcm = s->nf * kFrameSamples;
    if (cudaStreamCreateWithFlags(&s->host_stream, cudaStreamNonBlocking) != cudaSuccess ||
        cudaMallocHost(&s->h_codes, ncodes * 4) != cudaSuccess ||
        cudaMallocHost(&s->h_pcm, npcm * 4) != cudaSuccess ||
        cudaMalloc(&s->d_codes, ncodes * 4) != cudaSuccess ||
        cudaMalloc(&s->d_pcm, npcm * 4) != cudaSuccess)
      return fail(-2);
  }
  if (cudaDeviceSynchronize() != cudaSuccess) return fail(-2);
  // Streaming windows are cheap: pre-capture the graphs for F=4 at every B (noise on and
  // off) so the first window of any batch size pays no capture/instantiate cost.
  // PLOW_SNAC_PREWARM_F=<F> picks another window size, 0 disables.
  {
    int pf = 4;
    if (const char* e = getenv("PLOW_SNAC_PREWARM_F")) pf = atoi(e);
    if (s->use_graph && pf > 0 && pf <= max_frames) {
      for (int b = 1; b <= max_batch; ++b)
        for (int nz = 0; nz < 2; ++nz) {
          GraphEntry ge;
          if (int r = build_graph(s, b, pf, nz != 0, &ge)) return fail(r);
          s->graphs.emplace(std::make_tuple(b, pf, nz), ge);
        }
    }
  }
  *out = s;
  return 0;
}

extern "C" int plow_snac_destroy(void* hp) {
  if (!hp) return -1;
  Snac* s = (Snac*)hp;
  cudaSetDevice(s->device);
  cudaDeviceSynchronize();
  for (auto& kv : s->graphs) {
    if (kv.second.exec) cudaGraphExecDestroy(kv.second.exec);
    if (kv.second.graph) cudaGraphDestroy(kv.second.graph);
  }
  if (s->cap_stream) cudaStreamDestroy(s->cap_stream);
  if (s->host_stream) cudaStreamDestroy(s->host_stream);
  cudaFreeHost(s->h_codes);
  cudaFreeHost(s->h_pcm);
  cudaFree(s->d_codes);
  cudaFree(s->d_pcm);
  cudaFree(s->cnt);
  cudaFree(s->ws);
  cudaFree(s->seed_slot);
  cudaFree(s->xC);
  cudaFree(s->xB);
  cudaFree(s->xA);
  cudaFree(s->wsplit);
  cudaFree(s->wbuf);
  delete s;
  return 0;
}

extern "C" int plow_snac_decode(void* hp, const int32_t* codes, int B, int F, float* pcm_out,
                                unsigned long long noise_seed, void* stream) {
  if (!hp || !codes || !pcm_out || B < 1 || F < 1) return -1;
  Snac* s = (Snac*)hp;
  if (B > s->max_batch || F > s->max_frames) return -6;
  cudaStream_t st = (cudaStream_t)stream;
  if (!s->use_graph) return enqueue(s, codes, B, F, pcm_out, noise_seed, st);
  auto key = std::make_tuple(B, F, noise_seed ? 1 : 0);
  auto it = s->graphs.find(key);
  if (it == s->graphs.end()) {
    GraphEntry ge;
    if (int r = build_graph(s, B, F, noise_seed != 0, &ge)) return r;
    it = s->graphs.emplace(key, ge).first;
  }
  GraphEntry& ge = it->second;
  const int T = 4 * F;
  QuantArgs qa{codes, B, F, s->P[0], s->P[1], s->P[2], s->pre_dw_w, s->pre_dw_b, s->xC,
               noise_seed, s->seed_slot};
  void* qargs[] = {&qa};
  cudaKernelNodeParams qp{};
  qp.func = (void*)k_quant;
  qp.gridDim = dim3(blocks((long)B * T * kLat, 256));
  qp.blockDim = dim3(256);
  qp.kernelParams = qargs;
  CK(cudaGraphExecKernelNodeSetParams(ge.exec, ge.quant_node, &qp));
  const int Tout = F * kFrameSamples;
  FinalArgs fa{ge.final_in, Tout, s->out_snake, s->out_w, s->out_b, pcm_out};
  void* fargs[] = {&fa};
  cudaKernelNodeParams fp{};
  fp.func = (void*)k_final;
  fp.gridDim = dim3(blocks(Tout, FIN_TT), B);
  fp.blockDim = dim3(FIN_TT);
  fp.kernelParams = fargs;
  CK(cudaGraphExecKernelNodeSetParams(ge.exec, ge.final_node, &fp));
  CK(cudaGraphLaunch(ge.exec, st));
  return 0;
}

// Host-memory convenience entry for a plain CPU thread (no CUDA state of its own): selects
// the handle's device (primary context), stages codes through pinned memory, runs the same
// graph-cached decode on the handle's internal stream, copies PCM back and synchronizes
// that stream before returning. Must not run concurrently with another decode on the
// same handle.
extern "C" int plow_snac_decode_host(void* hp, const int32_t* codes_host, int B, int F,
                                     float* pcm_host, unsigned long long noise_seed) {
  if (!hp || !codes_host || !pcm_host || B < 1 || F < 1) return -1;
  Snac* s = (Snac*)hp;
  if (B > s->max_batch || F > s->max_frames) return -6;
  CK(cudaSetDevice(s->device));
  const size_t ncodes = (size_t)B * F * 7, npcm = (size_t)B * F * kFrameSamples;
  memcpy(s->h_codes, codes_host, ncodes * 4);
  CK(cudaMemcpyAsync(s->d_codes, s->h_codes, ncodes * 4, cudaMemcpyHostToDevice, s->host_stream));
  if (int r = plow_snac_decode(s, s->d_codes, B, F, s->d_pcm, noise_seed, s->host_stream)) return r;
  CK(cudaMemcpyAsync(s->h_pcm, s->d_pcm, npcm * 4, cudaMemcpyDeviceToHost, s->host_stream));
  CK(cudaStreamSynchronize(s->host_stream));
  memcpy(pcm_host, s->h_pcm, npcm * 4);
  return 0;
}

// Test hook (not part of the plowrt ABI): copies n floats of the activation captured by
// PLOW_SNAC_DEBUG_STOP into dst (device) on `stream`.
extern "C" int plow_snac_debug_read(void* hp, float* dst, long n, void* stream) {
  Snac* s = (Snac*)hp;
  if (!s || !s->dbg_out) return -1;
  CK(cudaMemcpyAsync(dst, s->dbg_out, n * 4, cudaMemcpyDeviceToDevice, (cudaStream_t)stream));
  return 0;
}
