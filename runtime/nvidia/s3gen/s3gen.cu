// libplow_s3gen.so: native Chatterbox S3Gen (S3 speech tokens -> 24 kHz PCM) for plowrt.
//
// Implements S3Token2Wav.inference(speech_tokens, ref_dict) of chatterbox-tts 0.1.7 (English
// s3gen.safetensors, non-meanflow): token embedding + UpsampleConformerEncoder (6 rel-pos
// conformer blocks, nearest-x2 + conv upsample, 4 more blocks) + encoder_proj; speaker x-vector
// affine; CausalConditionalCFM, 10 Euler steps on the cosine schedule, CFG rate 0.7 (batch
// cond|uncond) with the ConditionalDecoder estimator (1 down + 12 mid + 1 up causal ResNet1D
// blocks, 56 transformer blocks 8 heads x 64); prompt-mel conditioning; HiFTGenerator.inference
// (ConvRNNF0Predictor, SineGen/NSF source, STFT n_fft 16 hop 4, conv-transpose 8/5/3 with snake
// ResBlocks, iSTFT) and the 40 ms trim_fade.
//
// C ABI (all functions return 0 on success, negative on error):
//   int plow_s3gen_create(int device, const char* weights_path, int max_batch, int max_tokens,
//                         void** out);
//   int plow_s3gen_destroy(void* h);
//   int plow_s3gen_add_voice(void* h, const char* voice_path, int* voice_id);
//   int plow_s3gen_synthesize_host(void* h, int voice_id, const int32_t* tokens, int n_tokens,
//                                  float* wav_out, int max_samples, unsigned long long seed,
//                                  int* n_samples_out);
//   int plow_s3gen_synthesize_batch_host(void* h, int B, const int* voice_ids,
//                                  const int32_t* tokens /*concatenated*/, const int* n_tokens,
//                                  float* wav_out /*[B][max_samples]*/, int max_samples,
//                                  const unsigned long long* seeds, int* n_samples_out /*[B]*/);
//   (blocking; own stream + pinned staging; callable from a plain CPU thread with no CUDA state;
//    one call at a time per handle.) n_samples = 480 * (2*(P + n_tokens) - Pf) where P / Pf are
//    the voice's prompt token / prompt mel lengths (P = 157, Pf = 314 for the builtin voice, so
//    n_samples = 960 * n_tokens). Tokens are clamped to 0..6560.
// Error codes: -1 bad argument, -2 CUDA error, -4 cannot read file, -5 bad file,
//              -6 exceeds create-time capacity (max_batch / max_tokens / prompt), -7 wav_out too
//              small (n_samples_out is still filled in).
//
// Files: the weight / voice blobs written by scripts/tts/s3gen_prep.py (format documented there).
//
// Randomness (torch RNG replaced by a counter-based generator; outputs are a pure function of
// (weights, voice, tokens, seed) and do not depend on the batch composition):
//   u = splitmix64-hash(seed, stream, a, b); normal = Box-Muller of two 24-bit uniforms.
//   stream 1: CFM initial noise z[t][c] (t = mel frame incl. the prompt, c = mel bin)
//   stream 2: SineGen initial phase of harmonic i = 1..8, Uniform(-pi, pi) (harmonic 0: 0)
//   stream 3: SineGen additive noise n_i[sample], N(0,1) * (voiced ? 0.003 : 0.1/3)
// The SineGen phase integral is accumulated in fp64 (torch: fp32 cumsum).
//
// Execution: every intermediate is channels-last fp32 [batch][row capacity][C]. Per-item lengths
// live in device memory (computed from the call arguments by the first kernel), so one CUDA graph
// serves every length in a capacity bucket (rows rounded up to 128 tokens / 128 mel frames); tiles
// wholly past an item's length exit at once. Graphs are captured on first use per (B, bucket) and
// cached; PLOW_S3GEN_GRAPH=0 launches directly. Launches per utterance: see plow_s3gen_stats.
// GEMMs (every Linear / Conv1d / ConvTranspose1d, as implicit-im2col GEMMs with fused
// LayerNorm / Mish / snake / leaky-ReLU prologues and bias / activation / residual epilogues) run
// on mma.sync TF32 tensor cores with fp32 accumulation; attention (incl. the ESPnet relative
// position term) is a flash-style TF32 kernel with fp32 online softmax.
// PLOW_S3GEN_PREC=3xtf32 selects split-TF32 (3 MMAs, fp32-level accuracy) for all GEMMs,
// "tf32" (default) one MMA per product.
#include <cuda_runtime.h>
#include <stdint.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace {

constexpr int kMaxB = 16;
constexpr int kLS = 2 * kMaxB;  // lens stride per space (CFG doubles the batch)
enum { SP_TOK, SP_MEL, SP_GEN, SP_H0, SP_H1, SP_H2, SP_WAV, SP_AUX, NSP };
constexpr int kVocab = 6561, kDE = 512, kDC = 256, kMel = 80;
constexpr int kSteps = 10, kUp = 480, kBK = 32;
constexpr int kPeRows = 9999, kPeCenter = 4999;

struct Len {
  int sp, mul, add;
};
__device__ __forceinline__ int lenof(const int* lens, Len l, int b) {
  return lens[l.sp * kLS + b] * l.mul + l.add;
}

struct CallArgs {
  int B;
  int n_tok[kMaxB], P[kMaxB], Pf[kMaxB], tok_off[kMaxB];
  const int* ptok[kMaxB];
  const float* pfeat[kMaxB];
  const float* spks[kMaxB];
  unsigned long long seed[kMaxB];
};

// ---------------------------------------------------------------------------------------
// device helpers
__device__ __forceinline__ void griddep_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }
__device__ __forceinline__ uint32_t su32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ uint32_t tf32r(float x) {
  uint32_t u;
  asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(u) : "f"(x));
  return u;
}
__device__ __forceinline__ float tf32f(float x) { return __uint_as_float(tf32r(x)); }
__device__ __forceinline__ void cp_async16(void* s, const void* g) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(su32(s)), "l"(g));
}
__device__ __forceinline__ void cp_async16_z(void* s, const void* g, bool ok) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(su32(s)), "l"(g), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_wait0() { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }
__device__ __forceinline__ void mma_tf32(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float mishf(float x) { return x * tanhf(log1pf(expf(x))); }
// sin(y)^2 has period pi: Cody-Waite reduce by pi to [-pi/2, pi/2], then the SFU sine.
__device__ __forceinline__ float sin2(float y) {
  float k = rintf(y * 0.318309886183790672f);
  float r = fmaf(k, -3.14159274101257324f, y);
  r = fmaf(k, 8.74227800037248e-08f, r);
  float s = __sinf(r);
  return s * s;
}
__device__ __forceinline__ unsigned long long mix64(unsigned long long z) {
  z += 0x9e3779b97f4a7c15ULL;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
__device__ __forceinline__ unsigned long long rhash(unsigned long long seed, unsigned stream, unsigned a,
                                                    unsigned b) {
  unsigned long long key = ((unsigned long long)stream << 56) ^ ((unsigned long long)a << 32) ^ b;
  return mix64(seed ^ mix64(key));
}
__device__ __forceinline__ float rnormal(unsigned long long seed, unsigned stream, unsigned a, unsigned b) {
  unsigned long long h1 = rhash(seed, stream, a, b), h2 = mix64(h1);
  float u1 = (float)((h1 >> 40) + 1) * (1.0f / 16777216.0f);  // (0,1]
  float u2 = (float)(h2 >> 40) * (1.0f / 16777216.0f);        // [0,1)
  return sqrtf(-2.0f * logf(u1)) * cospif(2.0f * u2);
}
__device__ __forceinline__ float runiform(unsigned long long seed, unsigned stream, unsigned a, unsigned b) {
  return (float)(rhash(seed, stream, a, b) >> 40) * (1.0f / 16777216.0f);  // [0,1)
}

// ---------------------------------------------------------------------------------------
// Implicit-GEMM conv / linear on TF32 tensor cores.
//   out[b, to] = epi( sum_{tap, ci} pre(A[b', u >> shift][ci]) * W[z][n][tap*cin + ci] )
//   GEMM row t (item b, valid t < rlen), u = t*stride + tap*dil - pad (zero outside [0, alen)),
//   b' = b % a_bmod (a_bmod > 0: the CFG uncond half reads the cond half's input),
//   to = t*ostride + ooff + z (z = convT output phase), stored when o_lo <= to < olen.
//   epi: v = act(acc + bias); v += aux; v *= oscale; v += aux2   (aux/aux2 at output row to)
enum { PRE_NONE, PRE_LRELU, PRE_SNAKE, PRE_LN, PRE_LN_MISH, PRE_LN_MISH_ADD };
enum { ACT_NONE, ACT_SILU, ACT_GELU, ACT_ELU, ACT_ABS, ACT_LRELU };

struct GemmArgs {
  const int* lens;
  int tcap;
  Len rlen;
  const float* A;
  int lda, a_tcap, a_bmod;
  Len alen;
  int cin, K, kpad, stride, dil, pad, shift;
  int pre;
  const float *pg, *pb, *pt;
  float slope, eps;
  const float* W;
  long wz;
  int N;
  const float* bias;
  int act;
  float aslope;
  const float* aux;
  int ldaux;
  float oscale;
  const float* aux2;
  int ldaux2;
  float* out;
  int ldo, o_tcap, ostride, ooff, o_lo;
  Len olen;
  int reflect;  // also store output row 2 at row 0 (HiFT ReflectionPad1d((1, 0)) after a +1 shift)
};

// Tile configuration: BM x BN output tile, WM x WN per warp, KS warp groups splitting each
// stage's K (8 warps even for small tiles), STAGES-deep cp.async pipeline, BK = 32 * KS.
template <int BM_, int BN_, int WM_, int WN_, int KS_>
struct GCfg {
  static constexpr int BM = BM_, BN = BN_, WM = WM_, WN = WN_, KS = KS_, STAGES = 3;
  static constexpr int NWG = (BM / WM) * (BN / WN), NW = NWG * KS, NT = NW * 32;
  static constexpr int BK = 32 * KS, BKP = BK + 4;
  static constexpr int STAGE = (BM + BN) * BKP;  // floats per stage
  static constexpr int CP = BN + 4;              // epilogue tile pitch
  static constexpr int PIPE = STAGES * STAGE, EPI = KS * BM * CP;
  static constexpr int bytes = (PIPE > EPI ? PIPE : EPI) * 4 + (BM + 16) * 8;
};

template <int PRE>
__device__ __forceinline__ float4 pre_op(const GemmArgs& a, float4 v, int ci, int si, const float2* st) {
  switch (PRE) {
    case PRE_LRELU: {
      const float s = a.slope;
      v.x = v.x > 0.f ? v.x : v.x * s;
      v.y = v.y > 0.f ? v.y : v.y * s;
      v.z = v.z > 0.f ? v.z : v.z * s;
      v.w = v.w > 0.f ? v.w : v.w * s;
      return v;
    }
    case PRE_SNAKE: {
      const float4 al = __ldg(reinterpret_cast<const float4*>(a.pg + ci));
      v.x = fmaf(1.0f / (al.x + 1e-9f), sin2(v.x * al.x), v.x);
      v.y = fmaf(1.0f / (al.y + 1e-9f), sin2(v.y * al.y), v.y);
      v.z = fmaf(1.0f / (al.z + 1e-9f), sin2(v.z * al.z), v.z);
      v.w = fmaf(1.0f / (al.w + 1e-9f), sin2(v.w * al.w), v.w);
      return v;
    }
    case PRE_LN:
    case PRE_LN_MISH:
    case PRE_LN_MISH_ADD: {
      const float2 s = st[si];
      const float4 g = __ldg(reinterpret_cast<const float4*>(a.pg + ci));
      const float4 bb = __ldg(reinterpret_cast<const float4*>(a.pb + ci));
      v.x = (v.x - s.x) * s.y * g.x + bb.x;
      v.y = (v.y - s.x) * s.y * g.y + bb.y;
      v.z = (v.z - s.x) * s.y * g.z + bb.z;
      v.w = (v.w - s.x) * s.y * g.w + bb.w;
      if (PRE != PRE_LN) {
        v.x = mishf(v.x);
        v.y = mishf(v.y);
        v.z = mishf(v.z);
        v.w = mishf(v.w);
      }
      if (PRE == PRE_LN_MISH_ADD) {
        const float4 t = __ldg(reinterpret_cast<const float4*>(a.pt + ci));
        v.x += t.x;
        v.y += t.y;
        v.z += t.z;
        v.w += t.w;
      }
      return v;
    }
    default:
      return v;
  }
}

__device__ __forceinline__ float act_op(int act, float v, float slope) {
  switch (act) {
    case ACT_SILU:
      return v / (1.0f + expf(-v));
    case ACT_GELU:
      return 0.5f * v * (1.0f + erff(v * 0.70710678118654752f));
    case ACT_ELU:
      return v > 0.f ? v : expm1f(v);
    case ACT_ABS:
      return fabsf(v);
    case ACT_LRELU:
      return v > 0.f ? v : v * slope;
    default:
      return v;
  }
}

template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

// Kernels are specialised on the prologue kind (PRE) so each instance carries one activation
// path. Both operands stream through a STAGES-deep cp.async ring (A with zero fill for conv
// padding / sequence ends / K padding); pre-ops are applied in shared memory by the thread that
// loaded the chunk, once it has landed. The epilogue stages the (K-group summed) tile through
// shared memory and runs one small non-unrolled loop (row-contiguous, coalesced).
template <class C, int PREC, int PRE>
__global__ void __launch_bounds__(C::NT) k_gemm(const __grid_constant__ GemmArgs a) {
  constexpr int BM = C::BM, BN = C::BN, WM = C::WM, WN = C::WN, KS = C::KS, NT = C::NT, NWG = C::NWG;
  constexpr int BK = C::BK, BKP = C::BKP, ST = C::STAGES;
  constexpr int MT = WM / 16, NT8 = WN / 8, NWN = BN / WN;
  constexpr int CPR = BK / 4;  // 16-byte chunks per tile row
  constexpr int AV = BM * CPR / NT, BV = BN * CPR / NT;
  static_assert(BM * CPR % NT == 0 && BN * CPR % NT == 0, "chunking");
  extern __shared__ float4 gsm4[];
  float* sm = reinterpret_cast<float*>(gsm4);
  float2* st = reinterpret_cast<float2*>(sm + (C::PIPE > C::EPI ? C::PIPE : C::EPI));

  const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN, z = blockIdx.z;
  const int b = m0 / a.tcap, t0 = m0 - b * a.tcap;
  griddep_wait();
  const int rlen = lenof(a.lens, a.rlen, b);
  if (t0 >= rlen) return;
  const int bi = a.a_bmod ? b % a.a_bmod : b;
  const int alen = lenof(a.lens, a.alen, bi);
  const float* Ab = a.A + (long)bi * a.a_tcap * a.lda;
  const float* W = a.W + (long)z * a.wz + (long)n0 * a.kpad;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int u_lo = t0 * a.stride - a.pad;

  if (PRE >= PRE_LN) {  // per-row LayerNorm statistics (two-pass, fp32) incl. conv halo rows
    const int nst = (BM - 1) * a.stride + (a.K / a.cin - 1) * a.dil + 1;
    const int nv = a.cin >> 5;
    for (int r = warp; r < nst; r += C::NW) {
      const int u = u_lo + r;
      float2 s = make_float2(0.f, 0.f);
      if (u >= 0 && u < alen) {
        const float* row = Ab + (long)(u >> a.shift) * a.lda;
        float v[16];
        float sum = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i)
          if (i < nv) {
            v[i] = row[i * 32 + lane];
            sum += v[i];
          }
        const float mean = warp_sum(sum) / a.cin;
        float sq = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i)
          if (i < nv) {
            const float d = v[i] - mean;
            sq += d * d;
          }
        s = make_float2(mean, 1.0f / sqrtf(warp_sum(sq) / a.cin + a.eps));
      }
      if (lane == 0) st[r] = s;
    }
    __syncthreads();
  }

  // A chunk (row r, 4 channels at k): source row u, validity. meta = ci | (u - u_lo) << 16, or -1.
  auto a_meta = [&](int k, int r, const float** src) -> int {
    if (k >= a.K) return -1;
    const int tap = k / a.cin, ci = k - tap * a.cin;
    const int u = (t0 + r) * a.stride + tap * a.dil - a.pad;
    if (u < 0 || u >= alen) return -1;
    *src = Ab + (long)(u >> a.shift) * a.lda + ci;
    return ci | ((PRE >= PRE_LN ? u - u_lo : 0) << 16);
  };
  auto issue = [&](int kt) {
    const int s = kt % ST, k0 = kt * BK;
    float* As = sm + s * C::STAGE;
    float* Bs = As + BM * BKP;
#pragma unroll
    for (int i = 0; i < AV; ++i) {
      const int idx = tid + i * NT, r = idx / CPR, c4 = idx % CPR;
      const float* src = Ab;
      const int m = a_meta(k0 + c4 * 4, r, &src);
      cp_async16_z(As + r * BKP + c4 * 4, src, m >= 0);
    }
#pragma unroll
    for (int i = 0; i < BV; ++i) {
      const int idx = tid + i * NT, r = idx / CPR, c4 = idx % CPR;
      cp_async16(Bs + r * BKP + c4 * 4, W + (long)r * a.kpad + k0 + c4 * 4);
    }
  };
  auto transform = [&](int kt) {  // pre-op on this thread's own (landed) A chunks
    const int s = kt % ST, k0 = kt * BK;
    float* As = sm + s * C::STAGE;
#pragma unroll
    for (int i = 0; i < AV; ++i) {
      const int idx = tid + i * NT, r = idx / CPR, c4 = idx % CPR;
      const float* src;
      const int m = a_meta(k0 + c4 * 4, r, &src);
      if (m >= 0) {
        float4* p = reinterpret_cast<float4*>(As + r * BKP + c4 * 4);
        *p = pre_op<PRE>(a, *p, m & 0xffff, m >> 16, st);
      }
    }
  };

  const int kg = warp / NWG, wig = warp - kg * NWG;  // K group, warp in group
  const int wm = wig / NWN, wn = wig - wm * NWN, g = lane >> 2, tq = lane & 3;
  float acc[MT][NT8][4];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NT8; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

  const int nk = a.kpad / BK;
#pragma unroll
  for (int s = 0; s < ST - 1; ++s) {
    if (s < nk) issue(s);
    cp_commit();
  }
  for (int kt = 0; kt < nk; ++kt) {
    cp_wait<ST - 2>();
    if (PRE != PRE_NONE) transform(kt);
    __syncthreads();
    if (kt + ST - 1 < nk) issue(kt + ST - 1);
    cp_commit();
    const float* As = sm + (kt % ST) * C::STAGE;
    const float* Asb = As + (wm * WM) * BKP + kg * 32;
    const float* Bsb = As + BM * BKP + (wn * WN) * BKP + kg * 32;
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      uint32_t af[MT][4], bf[NT8][2];
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        const float* p = Asb + (mt * 16 + g) * BKP + kk * 8 + tq;
        af[mt][0] = __float_as_uint(p[0]);
        af[mt][1] = __float_as_uint(p[8 * BKP]);
        af[mt][2] = __float_as_uint(p[4]);
        af[mt][3] = __float_as_uint(p[8 * BKP + 4]);
      }
#pragma unroll
      for (int nt = 0; nt < NT8; ++nt) {
        const float* p = Bsb + (nt * 8 + g) * BKP + kk * 8 + tq;
        bf[nt][0] = __float_as_uint(p[0]);
        bf[nt][1] = __float_as_uint(p[4]);
      }
      if (PREC == 1) {
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
          for (int e = 0; e < 4; ++e) af[mt][e] = tf32r(__uint_as_float(af[mt][e]));
#pragma unroll
        for (int nt = 0; nt < NT8; ++nt) {
          bf[nt][0] = tf32r(__uint_as_float(bf[nt][0]));
          bf[nt][1] = tf32r(__uint_as_float(bf[nt][1]));
        }
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
          for (int nt = 0; nt < NT8; ++nt) mma_tf32(acc[mt][nt], af[mt], bf[nt][0], bf[nt][1]);
      } else {
        uint32_t ah[MT][4], al[MT][4], bh[NT8][2], bl[NT8][2];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const float x = __uint_as_float(af[mt][e]);
            ah[mt][e] = tf32r(x);
            al[mt][e] = tf32r(x - __uint_as_float(ah[mt][e]));
          }
#pragma unroll
        for (int nt = 0; nt < NT8; ++nt)
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const float x = __uint_as_float(bf[nt][e]);
            bh[nt][e] = tf32r(x);
            bl[nt][e] = tf32r(x - __uint_as_float(bh[nt][e]));
          }
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
          for (int nt = 0; nt < NT8; ++nt) {
            mma_tf32(acc[mt][nt], al[mt], bh[nt][0], bh[nt][1]);
            mma_tf32(acc[mt][nt], ah[mt], bl[nt][0], bl[nt][1]);
            mma_tf32(acc[mt][nt], ah[mt], bh[nt][0], bh[nt][1]);
          }
      }
    }
  }
  cp_wait<0>();
  __syncthreads();  // all warps done with the pipeline buffers

  // ---- epilogue: per-K-group tiles in smem, summed in group order (deterministic) ----
  constexpr int CP = C::CP;
  float* Cs = sm + kg * BM * CP;
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int nt = 0; nt < NT8; ++nt)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh)
        *reinterpret_cast<float2*>(Cs + (wm * WM + mt * 16 + g + 8 * hh) * CP + wn * WN + nt * 8 + 2 * tq) =
            make_float2(acc[mt][nt][2 * hh], acc[mt][nt][2 * hh + 1]);
  __syncthreads();

  const int olen = lenof(a.lens, a.olen, b);
  const bool pair_ok = !(a.ldo & 1) && !(a.aux && (a.ldaux & 1)) && !(a.aux2 && (a.ldaux2 & 1));
  constexpr int HB = BN / 2;
#pragma unroll 1
  for (int idx = tid; idx < BM * HB; idx += NT) {
    const int r = idx / HB, c = (idx - r * HB) * 2;
    const int t = t0 + r, n = n0 + c;
    if (t >= rlen || n >= a.N) continue;
    const int to = t * a.ostride + a.ooff + z;
    if (to < a.o_lo || to >= olen) continue;
    const bool two = n + 1 < a.N;
    float v0 = sm[r * CP + c], v1 = sm[r * CP + c + 1];
#pragma unroll
    for (int q = 1; q < KS; ++q) {
      v0 += sm[q * BM * CP + r * CP + c];
      v1 += sm[q * BM * CP + r * CP + c + 1];
    }
    if (a.bias) {
      v0 += __ldg(a.bias + n);
      if (two) v1 += __ldg(a.bias + n + 1);
    }
    v0 = act_op(a.act, v0, a.aslope);
    v1 = act_op(a.act, v1, a.aslope);
    const bool dup = a.reflect && to == 2;
#pragma unroll 1
    for (int rep = 0; rep < (dup ? 2 : 1); ++rep) {
      const long rr = (long)b * a.o_tcap + (rep ? 0 : to);
      float w0 = v0, w1 = v1;
      if (a.aux) {
        const float* ap = a.aux + rr * a.ldaux + n;
        w0 += ap[0];
        if (two) w1 += ap[1];
      }
      w0 *= a.oscale;
      w1 *= a.oscale;
      if (a.aux2) {
        const float* ap = a.aux2 + rr * a.ldaux2 + n;
        w0 += ap[0];
        if (two) w1 += ap[1];
      }
      float* op = a.out + rr * a.ldo + n;
      if (two && pair_ok) {
        *reinterpret_cast<float2*>(op) = make_float2(w0, w1);
      } else {
        op[0] = w0;
        if (two) op[1] = w1;
      }
    }
  }
}

// ---- Hopper wgmma TF32 (K-major operands, 128-byte swizzle; recipe of ../snac/snac.cu) -------
// Tile row = 32 fp32 = 128 B = one swizzle atom row; 16-byte chunk c of row r lives at float
// offset r*32 + ((c ^ (r & 7)) * 4). Descriptor LBO 16 B, SBO 1024 B, swizzle 128 B; a k8
// substep advances the start address by 32 B. Atoms are 1024-byte aligned. wgmma reads fp32 bit
// patterns and ignores the low 13 bits, so operands are stored pre-rounded (RN) to TF32.
__device__ __forceinline__ uint64_t wg_desc(const float* p) {
  uint64_t a = (uint64_t)__cvta_generic_to_shared(p);
  return ((a & 0x3FFFFull) >> 4) | ((16ull >> 4) << 16) | ((1024ull >> 4) << 32) | (1ull << 62);
}
__device__ __forceinline__ int swz(int row, int c4) { return row * 32 + ((c4 ^ (row & 7)) << 2); }
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wg_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}
__device__ __forceinline__ void fence_async_smem() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_n128(float* d, uint64_t da, uint64_t db) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k8.f32.tf32.tf32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, p, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
      : "l"(da), "l"(db));
}
__device__ __forceinline__ void wgmma_n64(float* d, uint64_t da, uint64_t db) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k8.f32.tf32.tf32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, p, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "l"(da), "l"(db));
}

// Shared epilogue over a staged tile Cs[KS][BM][BN + 4] (KS partial tiles summed in order):
// v = act(acc + bias); v += aux; v *= oscale; v += aux2; stored at output row to.
// Four independent pairs per thread per round, loads batched ahead of the math and stores.
template <int BM, int BN, int KS, int NT>
__device__ __forceinline__ void gemm_epilogue(const GemmArgs& a, const float* Cs, int b, int t0, int n0, int z,
                                              int rlen, int tid) {
  constexpr int CP = BN + 4, HB = BN / 2, U = 4;
  const int olen = lenof(a.lens, a.olen, b);
  const bool pair_ok = !(a.ldo & 1) && !(a.aux && (a.ldaux & 1)) && !(a.aux2 && (a.ldaux2 & 1));
  const long obase = (long)b * a.o_tcap;
#pragma unroll 1
  for (int base = tid; base < BM * HB; base += NT * U) {
    float v0[U], v1[U], x0[U], x1[U], y0[U], y1[U];
    int n[U], to[U];
    bool ok[U];
#pragma unroll
    for (int j = 0; j < U; ++j) {
      const int idx = base + j * NT;
      const int r = idx / HB, c = (idx - r * HB) * 2;
      const int t = t0 + r;
      n[j] = n0 + c;
      to[j] = t * a.ostride + a.ooff + z;
      ok[j] = idx < BM * HB && t < rlen && n[j] < a.N && to[j] >= a.o_lo && to[j] < olen;
      const int rc = min(idx, BM * HB - 1);
      const int r2 = rc / HB, c2 = (rc - r2 * HB) * 2;
      v0[j] = Cs[r2 * CP + c2];
      v1[j] = Cs[r2 * CP + c2 + 1];
#pragma unroll
      for (int q = 1; q < KS; ++q) {
        v0[j] += Cs[q * BM * CP + r2 * CP + c2];
        v1[j] += Cs[q * BM * CP + r2 * CP + c2 + 1];
      }
    }
#pragma unroll
    for (int j = 0; j < U; ++j) {
      x0[j] = x1[j] = y0[j] = y1[j] = 0.f;
      if (!ok[j]) continue;
      const bool two = n[j] + 1 < a.N;
      if (a.bias) {
        v0[j] += __ldg(a.bias + n[j]);
        if (two) v1[j] += __ldg(a.bias + n[j] + 1);
      }
      const long rr = obase + to[j];
      if (a.aux) {
        const float* ap = a.aux + rr * a.ldaux + n[j];
        x0[j] = ap[0];
        if (two) x1[j] = ap[1];
      }
      if (a.aux2) {
        const float* ap = a.aux2 + rr * a.ldaux2 + n[j];
        y0[j] = ap[0];
        if (two) y1[j] = ap[1];
      }
    }
#pragma unroll
    for (int j = 0; j < U; ++j) {
      if (!ok[j]) continue;
      const bool two = n[j] + 1 < a.N;
      const float w0 = act_op(a.act, v0[j], a.aslope), w1 = act_op(a.act, v1[j], a.aslope);
      const float o0 = (w0 + x0[j]) * a.oscale + y0[j], o1 = (w1 + x1[j]) * a.oscale + y1[j];
      float* op = a.out + (obase + to[j]) * a.ldo + n[j];
      if (two && pair_ok) {
        *reinterpret_cast<float2*>(op) = make_float2(o0, o1);
      } else {
        op[0] = o0;
        if (two) op[1] = o1;
      }
      if (a.reflect && to[j] == 2) {  // ReflectionPad1d((1, 0)): row 0 = row 2, with row 0's aux
        const float* ap = a.aux ? a.aux + obase * a.ldaux + n[j] : nullptr;
        const float* bp = a.aux2 ? a.aux2 + obase * a.ldaux2 + n[j] : nullptr;
        float* o = a.out + obase * a.ldo + n[j];
        o[0] = (w0 + (ap ? ap[0] : 0.f)) * a.oscale + (bp ? bp[0] : 0.f);
        if (two) o[1] = (w1 + (ap ? ap[1] : 0.f)) * a.oscale + (bp ? bp[1] : 0.f);
      }
    }
  }
}

// wgmma GEMM (default TF32 path), 256 threads = 2 warpgroups.
//   KSPLIT = 2: BM = 64; each stage holds 2 K-atoms of 32 and warpgroup w multiplies atom w
//               (two partial accumulators, summed in the epilogue) -> 8 warps even for small M.
//   KSPLIT = 1: BM = 128; one K-atom per stage, warpgroup w owns rows 64w..64w+63.
// ST-stage cp.async ring, prefetch distance ST - 2 (a slot is refilled only after the wgmma group
// that read it retired on every warpgroup). Each thread owns one fixed 16-byte column chunk of
// the rows it loads, so the implicit-conv source (tap, channel) advances incrementally. After a
// stage lands, each thread applies the pre-op and TF32 rounding to its own A chunks (loads
// batched ahead of stores), then fence.proxy.async + barrier publish the stage to the tensor
// cores. Weights are pre-rounded to TF32 at create.
template <int BN_, int KSPLIT_>
struct WCfg {
  static constexpr int BN = BN_, KSPLIT = KSPLIT_, BM = KSPLIT == 2 ? 64 : 128, NA = KSPLIT == 2 ? 2 : 1;
  static constexpr int ST = 4, NT = 256, BK = 32 * NA;
  static constexpr int ASZ = NA * BM * 32, BSZ = NA * BN * 32, STAGE = ASZ + BSZ;  // floats
  static constexpr int CP = BN + 4, PIPE = ST * STAGE, EPI = KSPLIT * BM * CP;
  static constexpr int bytes = (PIPE > EPI ? PIPE : EPI) * 4 + (BM + 16) * 8 + 1024;
};

template <class C, int PRE>
__global__ void __launch_bounds__(256) k_wgemm(const __grid_constant__ GemmArgs a) {
  constexpr int BM = C::BM, BN = C::BN, NA = C::NA, ST = C::ST, NT = C::NT, BK = C::BK, KSPLIT = C::KSPLIT;
  constexpr int CPR = NA * 8;      // 16-byte chunks per tile row per stage
  constexpr int RSTEP = NT / CPR;  // row step between one thread's chunks
  constexpr int AV = BM / RSTEP, BV = BN / RSTEP;
  static_assert(NT % CPR == 0 && BM % RSTEP == 0 && BN % RSTEP == 0, "chunking");
  extern __shared__ float4 wsm4[];
  float* sm = reinterpret_cast<float*>((reinterpret_cast<uintptr_t>(wsm4) + 1023) & ~uintptr_t(1023));
  float2* st = reinterpret_cast<float2*>(sm + (C::PIPE > C::EPI ? C::PIPE : C::EPI));

  const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN, z = blockIdx.z;
  const int b = m0 / a.tcap, t0 = m0 - b * a.tcap;
  griddep_wait();
  const int rlen = lenof(a.lens, a.rlen, b);
  if (t0 >= rlen) return;
  const int bi = a.a_bmod ? b % a.a_bmod : b;
  const int alen = lenof(a.lens, a.alen, bi);
  const float* Ab = a.A + (long)bi * a.a_tcap * a.lda;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int u_lo = t0 * a.stride - a.pad;

  if (PRE >= PRE_LN) {  // per-row LayerNorm statistics (two-pass, fp32) incl. conv halo rows
    const int nst = (BM - 1) * a.stride + (a.K / a.cin - 1) * a.dil + 1;
    const int nv = a.cin >> 5;
    for (int r = warp; r < nst; r += NT / 32) {
      const int u = u_lo + r;
      float2 s = make_float2(0.f, 0.f);
      if (u >= 0 && u < alen) {
        const float* row = Ab + (long)(u >> a.shift) * a.lda;
        float v[16];
        float sum = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i)
          if (i < nv) {
            v[i] = row[i * 32 + lane];
            sum += v[i];
          }
        const float mean = warp_sum(sum) / a.cin;
        float sq = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i)
          if (i < nv) {
            const float d = v[i] - mean;
            sq += d * d;
          }
        s = make_float2(mean, 1.0f / sqrtf(warp_sum(sq) / a.cin + a.eps));
      }
      if (lane == 0) st[r] = s;
    }
    __syncthreads();
  }

  // This thread's chunk column and rows.
  const int cc = tid % CPR, atom = cc >> 3, c4 = cc & 7, r0 = tid / CPR;
  int ci = 4 * cc, tap = 0;  // source channel / tap of this thread's k (= kt*BK + 4 cc)
  while (ci >= a.cin) {
    ci -= a.cin;
    ++tap;
  }
  int urow[AV];  // tap-independent source row of each owned A row
#pragma unroll
  for (int i = 0; i < AV; ++i) urow[i] = (t0 + r0 + i * RSTEP) * a.stride - a.pad;
  const float* Wp = a.W + (long)z * a.wz + (long)(n0 + r0) * a.kpad + 4 * cc;
  const long wstep = (long)RSTEP * a.kpad;
  auto issue = [&](int kt) {  // uses the current (tap, ci): called in kt order
    const int s = kt % ST;
    float* As = sm + s * C::STAGE + atom * BM * 32;
    float* Bs = sm + s * C::STAGE + C::ASZ + atom * BN * 32;
    const bool kok = kt * BK + 4 * cc < a.K;
    const int td = tap * a.dil;
#pragma unroll
    for (int i = 0; i < AV; ++i) {
      const int u = urow[i] + td;
      const bool ok = kok && u >= 0 && u < alen;
      const float* src = ok ? Ab + (long)(u >> a.shift) * a.lda + ci : Ab;
      cp_async16_z(As + swz(r0 + i * RSTEP, c4), src, ok);
    }
    const float* wsrc = Wp + kt * BK;
#pragma unroll
    for (int i = 0; i < BV; ++i) cp_async16(Bs + swz(r0 + i * RSTEP, c4), wsrc + i * wstep);
    ci += BK;
    while (ci >= a.cin) {
      ci -= a.cin;
      ++tap;
    }
  };
  // Pre-op + RN rounding of this thread's landed A chunks of stage kt.
  auto transform = [&](int kt) {
    const int k = kt * BK + 4 * cc;
    if (k >= a.K) return;
    float* As = sm + (kt % ST) * C::STAGE + atom * BM * 32;
    int tp = 0, c = k;
    if (PRE != PRE_NONE) {
      tp = k / a.cin;
      c = k - tp * a.cin;
    }
    const int td = tp * a.dil;
    float4 v[AV];
#pragma unroll
    for (int i = 0; i < AV; ++i) v[i] = *reinterpret_cast<const float4*>(As + swz(r0 + i * RSTEP, c4));
#pragma unroll
    for (int i = 0; i < AV; ++i) {
      if (PRE != PRE_NONE) {
        const int u = urow[i] + td;
        if (u >= 0 && u < alen) v[i] = pre_op<PRE>(a, v[i], c, u - u_lo, st);
      }
      v[i] = make_float4(tf32f(v[i].x), tf32f(v[i].y), tf32f(v[i].z), tf32f(v[i].w));
    }
#pragma unroll
    for (int i = 0; i < AV; ++i) *reinterpret_cast<float4*>(As + swz(r0 + i * RSTEP, c4)) = v[i];
  };

  constexpr int NACC = BN / 2;
  float acc[NACC];
#pragma unroll
  for (int i = 0; i < NACC; ++i) acc[i] = 0.f;
  const int wg = warp >> 2;
  const int nk = a.kpad / BK;
#pragma unroll
  for (int s = 0; s < ST - 2; ++s) {
    if (s < nk) issue(s);
    cp_commit();
  }
  for (int kt = 0; kt < nk; ++kt) {
    cp_wait<ST - 3>();
    transform(kt);
    fence_async_smem();
    __syncthreads();  // stage kt visible; every warpgroup has retired the wgmma of kt - 2
    if (kt + ST - 2 < nk) issue(kt + ST - 2);
    cp_commit();
    const float* stg = sm + (kt % ST) * C::STAGE;
    const float* As = stg + (KSPLIT == 2 ? wg * BM * 32 : wg * 64 * 32);
    const float* Bs = stg + C::ASZ + (KSPLIT == 2 ? wg * BN * 32 : 0);
    wg_fence();
#pragma unroll
    for (int at = 0; at < NA / KSPLIT; ++at)
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        const uint64_t da = wg_desc(As + at * BM * 32 + kk * 8), db = wg_desc(Bs + at * BN * 32 + kk * 8);
        if constexpr (BN == 128)
          wgmma_n128(acc, da, db);
        else
          wgmma_n64(acc, da, db);
      }
    wg_commit();
    wg_wait<1>();
  }
  wg_wait<0>();
  cp_wait<0>();
  __syncthreads();

  // accumulators -> smem. m64nBN layout, warp w of the warpgroup: reg 4g+2h+l ->
  // row 16w + lane/4 + 8h, col 8g + 2(lane%4) + l.
  constexpr int CP = C::CP;
  {
    const int w = warp & 3, rb = (KSPLIT == 2 ? 0 : wg * 64) + 16 * w + (lane >> 2), cb = 2 * (lane & 3);
    float* Cs = sm + (KSPLIT == 2 ? wg * BM * CP : 0);
#pragma unroll
    for (int gq = 0; gq < BN / 8; ++gq)
#pragma unroll
      for (int h = 0; h < 2; ++h)
        *reinterpret_cast<float2*>(Cs + (rb + 8 * h) * CP + 8 * gq + cb) =
            make_float2(acc[4 * gq + 2 * h], acc[4 * gq + 2 * h + 1]);
  }
  __syncthreads();
  gemm_epilogue<BM, BN, KSPLIT, NT>(a, sm, b, t0, n0, z, rlen, tid);
}

// ---------------------------------------------------------------------------------------
// Flash attention, 8 heads x 64, TF32 mma, fp32 online softmax. qkv rows [q(512) | k | v].
// REL: ESPnet rel-pos self-attention, scores = ((q + u)·k + (q + v)·P[i - j]) / 8, with the
// projected position table P (row r + rm - 1 <-> relative position r).
struct AttnArgs {
  const int* lens;
  Len len;
  const float* qkv;
  int ldq;
  float* out;
  int ldo;
  int tcap;
  const float* prel;
  int rm;
  const float* pu;
  const float* pv;
};
constexpr int AP = 68, BDP = 84;
template <bool REL>
struct ASmem {
  static constexpr int bytes = (3 * 64 * AP + (REL ? 128 * AP + 64 * BDP : 0)) * 4;
};

template <bool REL>
__global__ void __launch_bounds__(128) k_attn(const __grid_constant__ AttnArgs a) {
  extern __shared__ float4 asm4[];
  float* Ks = reinterpret_cast<float*>(asm4);
  float* Vs = Ks + 64 * AP;
  float* Ps = Vs + 64 * AP;
  float* Pr = Ps + 64 * AP;
  float* Bd = Pr + 128 * AP;
  const int h = blockIdx.y, b = blockIdx.z, i0 = blockIdx.x * 64;
  griddep_wait();
  const int len = lenof(a.lens, a.len, b);
  if (i0 >= len) return;
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, tq = lane & 3;
  const int iw = i0 + w * 16;
  const float* base = a.qkv + (long)b * a.tcap * a.ldq;
  uint32_t qa[8][4], qb[REL ? 8 : 1][4];
#pragma unroll
  for (int kk = 0; kk < 8; ++kk)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int row = iw + g + (e & 1) * 8, col = h * 64 + kk * 8 + tq + (e >> 1) * 4;
      const float q = base[(long)row * a.ldq + col];
      if (REL) {
        qa[kk][e] = tf32r((q + __ldg(a.pu + col)) * 0.125f);
        qb[kk % (REL ? 8 : 1)][e] = tf32r((q + __ldg(a.pv + col)) * 0.125f);
      } else {
        qa[kk][e] = tf32r(q * 0.125f);
      }
    }
  float o[8][4];
#pragma unroll
  for (int i = 0; i < 8; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
  float mrow[2] = {-INFINITY, -INFINITY}, lrow[2] = {0.f, 0.f};
  float* Pw = Ps + w * 16 * AP;
  for (int j0 = 0; j0 < len; j0 += 64) {
    __syncthreads();
    for (int i = tid; i < 64 * 16; i += 128) {
      const int r = i >> 4, c = (i & 15) * 4;
      const bool ok = j0 + r < len;
      const float* src = base + (long)(j0 + r) * a.ldq + 512 + h * 64 + c;
      cp_async16_z(Ks + r * AP + c, src, ok);
      cp_async16_z(Vs + r * AP + c, src + 512, ok);
    }
    if (REL) {
      const long r0 = (long)(i0 - j0 - 63) + a.rm - 1;
      for (int i = tid; i < 128 * 16; i += 128) {
        const int r = i >> 4, c = (i & 15) * 4;
        cp_async16(Pr + r * AP + c, a.prel + (r0 + r) * 512 + h * 64 + c);
      }
    }
    cp_commit();
    cp_wait0();
    __syncthreads();
    float s[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.f;
#pragma unroll
    for (int kk = 0; kk < 8; ++kk)
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        const float* p = Ks + (nt * 8 + g) * AP + kk * 8 + tq;
        mma_tf32(s[nt], qa[kk], tf32r(p[0]), tf32r(p[4]));
      }
    if (REL) {
      float bd[10][4];
#pragma unroll
      for (int i = 0; i < 10; ++i) bd[i][0] = bd[i][1] = bd[i][2] = bd[i][3] = 0.f;
#pragma unroll
      for (int kk = 0; kk < 8; ++kk)
#pragma unroll
        for (int nt = 0; nt < 10; ++nt) {
          const float* p = Pr + (w * 16 + nt * 8 + g) * AP + kk * 8 + tq;
          mma_tf32(bd[nt], qb[kk % (REL ? 8 : 1)], tf32r(p[0]), tf32r(p[4]));
        }
      float* Bw = Bd + w * 16 * BDP;
#pragma unroll
      for (int nt = 0; nt < 10; ++nt) {
        Bw[g * BDP + nt * 8 + 2 * tq] = bd[nt][0];
        Bw[g * BDP + nt * 8 + 2 * tq + 1] = bd[nt][1];
        Bw[(g + 8) * BDP + nt * 8 + 2 * tq] = bd[nt][2];
        Bw[(g + 8) * BDP + nt * 8 + 2 * tq + 1] = bd[nt][3];
      }
      __syncwarp();
#pragma unroll
      for (int nt = 0; nt < 8; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int gi = g + (e >> 1) * 8, jj = nt * 8 + 2 * tq + (e & 1);
          s[nt][e] += Bw[gi * BDP + gi - jj + 63];
        }
      __syncwarp();
    }
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e)
        if (j0 + nt * 8 + 2 * tq + (e & 1) >= len) s[nt][e] = -INFINITY;
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      float mx = -INFINITY;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) mx = fmaxf(mx, fmaxf(s[nt][2 * hh], s[nt][2 * hh + 1]));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
      const float mnew = fmaxf(mrow[hh], mx);
      const float alpha = expf(mrow[hh] - mnew);
      float sum = 0.f;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt)
#pragma unroll
        for (int e = 2 * hh; e < 2 * hh + 2; ++e) {
          const float p = expf(s[nt][e] - mnew);
          s[nt][e] = p;
          sum += p;
        }
      sum += __shfl_xor_sync(0xffffffffu, sum, 1);
      sum += __shfl_xor_sync(0xffffffffu, sum, 2);
      lrow[hh] = lrow[hh] * alpha + sum;
      mrow[hh] = mnew;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        o[nt][2 * hh] *= alpha;
        o[nt][2 * hh + 1] *= alpha;
      }
    }
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      *reinterpret_cast<float2*>(Pw + g * AP + nt * 8 + 2 * tq) = make_float2(tf32f(s[nt][0]), tf32f(s[nt][1]));
      *reinterpret_cast<float2*>(Pw + (g + 8) * AP + nt * 8 + 2 * tq) =
          make_float2(tf32f(s[nt][2]), tf32f(s[nt][3]));
    }
    __syncwarp();
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
      uint32_t pa[4];
      pa[0] = __float_as_uint(Pw[g * AP + kk * 8 + tq]);
      pa[1] = __float_as_uint(Pw[(g + 8) * AP + kk * 8 + tq]);
      pa[2] = __float_as_uint(Pw[g * AP + kk * 8 + tq + 4]);
      pa[3] = __float_as_uint(Pw[(g + 8) * AP + kk * 8 + tq + 4]);
#pragma unroll
      for (int nt = 0; nt < 8; ++nt)
        mma_tf32(o[nt], pa, tf32r(Vs[(kk * 8 + tq) * AP + nt * 8 + g]), tf32r(Vs[(kk * 8 + tq + 4) * AP + nt * 8 + g]));
    }
    __syncwarp();
  }
#pragma unroll
  for (int hh = 0; hh < 2; ++hh) {
    const int i = iw + g + 8 * hh;
    if (i >= len) continue;
    const float inv = 1.0f / lrow[hh];
    float* op = a.out + ((long)b * a.tcap + i) * a.ldo + h * 64 + 2 * tq;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
      *reinterpret_cast<float2*>(op + nt * 8) = make_float2(o[nt][2 * hh] * inv, o[nt][2 * hh + 1] * inv);
  }
}

// ---------------------------------------------------------------------------------------
// small kernels
__global__ void k_setup(const CallArgs* ca, int* lens) {
  const int b = threadIdx.x, B = ca->B;
  if (b >= B) return;
  const int T0 = ca->P[b] + ca->n_tok[b], T1 = 2 * T0, G = T1 - ca->Pf[b];
  lens[SP_TOK * kLS + b] = T0;
  lens[SP_MEL * kLS + b] = T1;
  lens[SP_MEL * kLS + b + B] = T1;
  lens[SP_GEN * kLS + b] = G;
  lens[SP_H0 * kLS + b] = 8 * G;
  lens[SP_H1 * kLS + b] = 40 * G;
  lens[SP_H2 * kLS + b] = 120 * G + 1;
  lens[SP_WAV * kLS + b] = kUp * G;
}

// Token embedding: prompt tokens then speech tokens.
__global__ void k_embed(const CallArgs* ca, const int* lens, const int* tokens, const float* emb, float* x,
                        int tcap) {
  const int t = blockIdx.x, b = blockIdx.y;
  griddep_wait();
  if (t >= lens[SP_TOK * kLS + b]) return;
  const int P = ca->P[b];
  int tok = t < P ? ca->ptok[b][t] : tokens[ca->tok_off[b] + t - P];
  tok = min(max(tok, 0), kVocab - 1);
  const float4* src = reinterpret_cast<const float4*>(emb + (long)tok * kDE);
  float4* dst = reinterpret_cast<float4*>(x + ((long)b * tcap + t) * kDE);
  dst[threadIdx.x] = src[threadIdx.x];
}

// y = (LayerNorm(x) * g + b) * scale, one warp per row, C <= 512 (multiple of 32).
__global__ void k_ln_rows(const float* x, int ldx, float* y, int ldy, const float* gg, const float* bb, float eps,
                          float scale, int C, int tcap, const int* lens, Len len, int nrows) {
  const int row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  griddep_wait();
  if (row >= nrows) return;
  const int b = row / tcap, t = row - b * tcap;
  if (t >= lenof(lens, len, b)) return;
  const float* xr = x + (long)row * ldx;
  const int nv = C >> 5;
  float v[16], sum = 0.f;
#pragma unroll
  for (int i = 0; i < 16; ++i)
    if (i < nv) {
      v[i] = xr[i * 32 + lane];
      sum += v[i];
    }
  const float mean = warp_sum(sum) / C;
  float sq = 0.f;
#pragma unroll
  for (int i = 0; i < 16; ++i)
    if (i < nv) sq += (v[i] - mean) * (v[i] - mean);
  const float rs = 1.0f / sqrtf(warp_sum(sq) / C + eps);
  float* yr = y + (long)row * ldy;
#pragma unroll
  for (int i = 0; i < 16; ++i)
    if (i < nv) {
      const int c = i * 32 + lane;
      yr[c] = ((v[i] - mean) * rs * gg[c] + bb[c]) * scale;
    }
}

// ResNet block tail: out = mish(LayerNorm(y2)) + r   (C = 256, eps 1e-5)
__global__ void k_resout(const float* y2, const float* r, const float* gg, const float* bb, float* out, int ldo,
                         int tcap, const int* lens, Len len, int nrows) {
  const int row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  griddep_wait();
  if (row >= nrows) return;
  const int b = row / tcap, t = row - b * tcap;
  if (t >= lenof(lens, len, b)) return;
  const float* yr = y2 + (long)row * kDC;
  float v[8], sum = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    v[i] = yr[i * 32 + lane];
    sum += v[i];
  }
  const float mean = warp_sum(sum) / kDC;
  float sq = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) sq += (v[i] - mean) * (v[i] - mean);
  const float rs = 1.0f / sqrtf(warp_sum(sq) / kDC + 1e-5f);
  const float* rr = r + (long)row * kDC;
  float* orow = out + (long)row * ldo;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int c = i * 32 + lane;
    orow[c] = mishf((v[i] - mean) * rs * gg[c] + bb[c]) + rr[c];
  }
}

// CFM inputs: x = z ~ N(0,1) (stream 1), ctx = [mu | spks | cond] (cond half), 0 (uncond half).
__global__ void k_cfm_init(const CallArgs* ca, const int* lens, const float* mu, float* x, float* z, float* ctx,
                           int tcap) {
  const int t = blockIdx.x, b = blockIdx.y, B = ca->B;
  griddep_wait();
  if (t >= lens[SP_MEL * kLS + b]) return;
  const long row = (long)b * tcap + t;
  const int Pf = ca->Pf[b];
  const unsigned long long seed = ca->seed[b];
  for (int c = threadIdx.x; c < 240; c += blockDim.x) {
    float v;
    if (c < 80)
      v = mu[row * kMel + c];
    else if (c < 160)
      v = ca->spks[b][c - 80];
    else
      v = t < Pf ? ca->pfeat[b][(long)t * kMel + c - 160] : 0.f;
    ctx[row * 240 + c] = v;
    ctx[((long)(b + B) * tcap + t) * 240 + c] = 0.f;
  }
  for (int c = threadIdx.x; c < kMel; c += blockDim.x) {
    const float v = rnormal(seed, 1, t, c);
    x[row * kMel + c] = v;
    z[row * kMel + c] = v;
  }
}

// x += dt * ((1 + 0.7) * d_cond - 0.7 * d_uncond)
__global__ void k_euler(const CallArgs* ca, const int* lens, float* x, const float* d, int tcap, float dt) {
  const int t = blockIdx.x, b = blockIdx.y, B = ca->B, c = threadIdx.x;
  griddep_wait();
  if (t >= lens[SP_MEL * kLS + b] || c >= kMel) return;
  const long row = (long)b * tcap + t;
  const float dc = d[row * kMel + c], du = d[((long)(b + B) * tcap + t) * kMel + c];
  x[row * kMel + c] = x[row * kMel + c] + dt * (1.7f * dc - 0.7f * du);
}

// Generated mel (prompt frames dropped): melg[b][t] = x[b][Pf + t]
__global__ void k_melgen(const CallArgs* ca, const int* lens, const float* x, int tcap1, float* melg, int tcapg) {
  const int t = blockIdx.x, b = blockIdx.y, c = threadIdx.x;
  griddep_wait();
  if (t >= lens[SP_GEN * kLS + b] || c >= kMel) return;
  melg[((long)b * tcapg + t) * kMel + c] = x[((long)b * tcap1 + ca->Pf[b] + t) * kMel + c];
}

// f0 frame prefix sums (fp64): pre[b][t] = sum_{t' < t} f0[b][t']
__global__ void k_f0scan(const int* lens, const float* f0, int tcapg, double* pre) {
  __shared__ double part[256];
  const int b = blockIdx.x, tid = threadIdx.x;
  griddep_wait();
  const int T = lens[SP_GEN * kLS + b];
  const int chunk = (T + 255) / 256, lo = tid * chunk, hi = min(T, lo + chunk);
  const float* f = f0 + (long)b * tcapg;
  double s = 0.0;
  for (int t = lo; t < hi; ++t) s += (double)f[t];
  part[tid] = s;
  __syncthreads();
  if (tid == 0) {
    double run = 0.0;
    for (int i = 0; i < 256; ++i) {
      const double v = part[i];
      part[i] = run;
      run += v;
    }
  }
  __syncthreads();
  double run = part[tid];
  double* p = pre + (long)b * (tcapg + 1);
  for (int t = lo; t < hi; ++t) {
    p[t] = run;
    run += (double)f[t];
  }
  if (hi == T && lo < hi) p[T] = run;
}

// NSF source: SineGen (9 harmonics, amp 0.1, voiced f0 > 10) + noise, l_linear, tanh.
__global__ void k_source(const CallArgs* ca, const int* lens, const float* f0, const double* pre, int tcapg,
                         const float* sw, const float* sb, float* src, int wavcap, float* dbg_noise,
                         float* dbg_phase) {
  const int b = blockIdx.y, n = blockIdx.x * blockDim.x + threadIdx.x;
  griddep_wait();
  const int L = lens[SP_WAV * kLS + b];
  if (n >= L) return;
  const int t = n / kUp, k = n - t * kUp;
  const float f = f0[(long)b * tcapg + t];
  const unsigned long long seed = ca->seed[b];
  const bool uv = f > 10.0f;
  const float amp = uv ? 0.003f : (float)(0.1 / 3.0);
  const double base = 480.0 * pre[(long)b * (tcapg + 1) + t] + (double)(k + 1) * (double)f;
  float acc = sb[0];
#pragma unroll
  for (int i = 0; i < 9; ++i) {
    double ph = base * (double)(i + 1) / 24000.0;
    ph -= floor(ph);
    const float pv = i == 0 ? 0.f : (runiform(seed, 2, i, 0) * 2.0f - 1.0f) * 3.14159265358979f;
    const float s = 0.1f * sinf((float)(6.283185307179586 * ph) + pv);
    const float nz = amp * rnormal(seed, 3, i, n);
    acc = fmaf(sw[i], (uv ? s : 0.f) + nz, acc);
    if (dbg_noise) dbg_noise[((long)b * 9 + i) * wavcap + n] = rnormal(seed, 3, i, n);
    if (dbg_phase && n == 0) dbg_phase[b * 9 + i] = pv;
  }
  src[(long)b * wavcap + n] = tanhf(acc);
}

__constant__ float c_cos16[16], c_sin16[16], c_win[16];

// STFT (n_fft 16, hop 4, Hann, center + reflect pad 8): out[b][f][0..8] = Re, [9..17] = Im, 18/19 = 0
__global__ void k_stft(const int* lens, const float* src, int wavcap, float* out, int h2cap) {
  const int b = blockIdx.y, f = blockIdx.x * blockDim.x + threadIdx.x;
  griddep_wait();
  if (f >= lens[SP_H2 * kLS + b]) return;
  const int L = lens[SP_WAV * kLS + b];
  const float* x = src + (long)b * wavcap;
  float xs[16];
#pragma unroll
  for (int n = 0; n < 16; ++n) {
    int p = 4 * f + n - 8;
    p = p < 0 ? -p : (p >= L ? 2 * (L - 1) - p : p);
    xs[n] = x[p] * c_win[n];
  }
  float* o = out + ((long)b * h2cap + f) * 20;
#pragma unroll
  for (int k = 0; k < 9; ++k) {
    float re = 0.f, im = 0.f;
#pragma unroll
    for (int n = 0; n < 16; ++n) {
      const int kn = (k * n) & 15;
      re = fmaf(xs[n], c_cos16[kn], re);
      im = fmaf(-xs[n], c_sin16[kn], im);
    }
    o[k] = re;
    o[9 + k] = im;
  }
  o[18] = 0.f;
  o[19] = 0.f;
}

// iSTFT of (exp(post[0..8]) clipped at 100, sin(post[9..17])), window-envelope normalised,
// clamp +-0.99, then trim_fade (first 480 samples zero, next 480 raised-cosine fade-in).
__constant__ float c_fade[480];
__global__ void k_istft(const int* lens, const float* post, int h2cap, float* wav, int wavcap) {
  const int b = blockIdx.y, s = blockIdx.x * blockDim.x + threadIdx.x;
  griddep_wait();
  const int L = lens[SP_WAV * kLS + b];
  if (s >= L) return;
  const int F = lens[SP_H2 * kLS + b];
  const int p = s + 8;
  const int flo = max(0, (p - 15 + 3) >> 2), fhi = min(F - 1, p >> 2);
  float y = 0.f, env = 0.f;
  for (int f = flo; f <= fhi; ++f) {
    const int n = p - 4 * f;
    const float* q = post + ((long)b * h2cap + f) * 18;
    float x = 0.f;
#pragma unroll
    for (int k = 0; k < 9; ++k) {
      const float mag = fminf(expf(q[k]), 100.0f);
      const float ph = sinf(q[9 + k]);
      float sp, cp;
      sincosf(ph, &sp, &cp);
      const int kn = (k * n) & 15;
      const float term = mag * cp * c_cos16[kn] - mag * sp * c_sin16[kn];
      x += (k == 0 || k == 8) ? (k == 0 ? mag * cp : mag * cp * c_cos16[kn]) : 2.0f * term;
    }
    x *= (1.0f / 16.0f);
    y = fmaf(c_win[n], x, y);
    env = fmaf(c_win[n], c_win[n], env);
  }
  y = env > 1e-11f ? y / env : y;
  y = fminf(fmaxf(y, -0.99f), 0.99f);
  if (s < 480)
    y = 0.f;
  else if (s < 960)
    y *= c_fade[s - 480];
  wav[(long)b * wavcap + s] = y;
}

// ---------------------------------------------------------------------------------------
// host side
#define CK(x)                                                                                 \
  do {                                                                                        \
    cudaError_t e_ = (x);                                                                     \
    if (e_ != cudaSuccess) {                                                                  \
      fprintf(stderr, "plow_s3gen: %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
      return -2;                                                                              \
    }                                                                                         \
  } while (0)
#define RC(x)                \
  do {                       \
    int r_ = (x);            \
    if (r_) return r_;       \
  } while (0)

inline unsigned cdiv(long n, long d) { return (unsigned)((n + d - 1) / d); }
inline int rup(int n, int d) { return (n + d - 1) / d * d; }

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

struct GW {  // padded device GEMM weight [phases][npad][kpad]; wr = the same rounded to TF32 (RN)
  const float* w = nullptr;
  const float* wr = nullptr;
  int N = 0, K = 0, npad = 0, kpad = 0, phases = 1;
};
struct TBW {
  const float *ln1g, *ln1b, *ln3g, *ln3b, *outb, *ff1b, *ff2b;
  GW qkv, out, ff1, ff2;
};
struct RNW {
  GW c1, c1c, r, rc, c2;
  const float *c1b, *rb, *ln1g, *ln1b, *c2b, *ln2g, *ln2b;
};
struct ELW {
  const float *lnmg, *lnmb, *qkvb, *pu, *pv, *outb, *lnfg, *lnfb, *ff1b, *ff2b;
  GW qkv, pos, out, ff1, ff2;
  float* prel = nullptr;
};
struct RBW {
  int k = 0;
  const float *a1[3], *a2[3], *c1b[3], *c2b[3];
  GW c1[3], c2[3];
};
struct Voice {
  int P = 0, Pf = 0;
  int* ptok = nullptr;
  float* pfeat = nullptr;
  float* spks = nullptr;
};
struct GraphEntry {
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  long launches = 0;
};
struct Caps {
  int B, t0, t1, g, h0, h1, h2, q0, q1, q2, wav;
};

struct Buf {
  float *e0, *e1, *e2, *qkv, *att, *ff, *mu, *x, *z, *ctx, *E1, *R1, *y1, *y2, *r, *h, *cat, *d;
  float *melg, *fa, *fb, *f0, *src, *stft, *hb[7], *post, *wav;
  double* f0p;
  float *dbg_noise, *dbg_phase;
};

struct S3 {
  int device = 0, max_batch = 0, max_tokens = 0, max_prompt = 0;
  int prec = 1;
  bool use_graph = true, debug = false;
  std::vector<char> blob;
  struct T {
    long off, nb;
    int dtype;
    std::vector<int> dims;
  };
  std::map<std::string, T> tens;
  float* wdev = nullptr;   // raw blob on device (vectors / tables)
  std::vector<void*> allocs;  // padded GEMM weights + tables
  // weights
  const float *emb, *emb_w_b, *emb_lng, *emb_lnb, *la1b, *la2b, *upb, *upe_b, *upe_lng, *upe_lnb, *aln_g, *aln_b,
      *projb;
  GW emb_w, la1, la2, up, upe_w, proj;
  ELW el[10];
  std::vector<float> spk_w, spk_b;  // host copy [80][192], [80]
  float tspan[kSteps + 1];
  const float* tvec;  // [10][14][256]
  RNW rn[14];
  TBW tb[56];
  GW downc, upc, fin, cproj;
  const float *downb, *upcb, *finb, *fin_lng, *fin_lnb, *cprojb;
  GW f0c[5], f0cls, hpre, hup[3], hsd[3], hpost;
  const float *f0cb[5], *f0clsb, *srcw, *srcb, *hpreb, *hupb[3], *hsdb[3], *hpostb;
  RBW sr[3], rb[9];
  // capacity
  int t0max = 0, gmax = 0, rm = 0;
  Buf buf{};
  int* d_lens = nullptr;
  CallArgs* d_call = nullptr;
  CallArgs* h_call = nullptr;
  int* d_tok = nullptr;
  int* h_tok = nullptr;
  float* h_wav = nullptr;
  cudaStream_t stream = nullptr, cap_stream = nullptr;
  std::vector<Voice> voices;
  std::map<std::tuple<int, int, int>, GraphEntry> graphs;
  long nlaunch = 0;  // launches enqueued by the current enqueue()
  long last_launches = 0, graph_launches = 0;
  Caps last_caps{};
};

// ---- GEMM launch ----
// Tile configs: 0 = 128x128 (8 warps, BK 32), 1 = 128x64 and 2 = 64x64 (8 warps as 2 K groups,
// BK 64).
using Cfg0 = GCfg<128, 128, 64, 32, 1>;
using Cfg1 = GCfg<128, 64, 64, 32, 2>;
using Cfg2 = GCfg<64, 64, 32, 32, 2>;
template <int CFG>
struct CfgOf;
template <>
struct CfgOf<0> {
  using T = Cfg0;
};
template <>
struct CfgOf<1> {
  using T = Cfg1;
};
template <>
struct CfgOf<2> {
  using T = Cfg2;
};
template <int CFG, int PREC, int PRE>
cudaError_t gemm_kernel(const GemmArgs* a, int M, int phases, cudaStream_t st) {
  using C = typename CfgOf<CFG>::T;
  auto k = k_gemm<C, PREC, PRE>;
  if (!a) return cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, C::bytes);
  dim3 grid(M / C::BM, cdiv(a->N, C::BN), phases);
  return launch(k, grid, dim3(C::NT), C::bytes, st, *a);
}
template <int CFG, int PREC>
cudaError_t gemm_pre(const GemmArgs* a, int pre, int M, int phases, cudaStream_t st) {
  switch (pre) {
    case PRE_NONE:
      return gemm_kernel<CFG, PREC, PRE_NONE>(a, M, phases, st);
    case PRE_LRELU:
      return gemm_kernel<CFG, PREC, PRE_LRELU>(a, M, phases, st);
    case PRE_SNAKE:
      return gemm_kernel<CFG, PREC, PRE_SNAKE>(a, M, phases, st);
    case PRE_LN:
      return gemm_kernel<CFG, PREC, PRE_LN>(a, M, phases, st);
    case PRE_LN_MISH:
      return gemm_kernel<CFG, PREC, PRE_LN_MISH>(a, M, phases, st);
    default:
      return gemm_kernel<CFG, PREC, PRE_LN_MISH_ADD>(a, M, phases, st);
  }
}
using WCfg0 = WCfg<128, 1>;  // 128 x 128
using WCfg1 = WCfg<128, 2>;  // 64 x 128, K split over the two warpgroups
using WCfg2 = WCfg<64, 2>;   // 64 x 64, K split
template <class C, int PRE>
cudaError_t wgemm_kernel(const GemmArgs* a, int M, int phases, cudaStream_t st) {
  auto k = k_wgemm<C, PRE>;
  if (!a) return cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, C::bytes);
  dim3 grid(M / C::BM, cdiv(a->N, C::BN), phases);
  return launch(k, grid, dim3(C::NT), C::bytes, st, *a);
}
template <class C>
cudaError_t wgemm_pre(const GemmArgs* a, int pre, int M, int phases, cudaStream_t st) {
  switch (pre) {
    case PRE_NONE:
      return wgemm_kernel<C, PRE_NONE>(a, M, phases, st);
    case PRE_LRELU:
      return wgemm_kernel<C, PRE_LRELU>(a, M, phases, st);
    case PRE_SNAKE:
      return wgemm_kernel<C, PRE_SNAKE>(a, M, phases, st);
    case PRE_LN:
      return wgemm_kernel<C, PRE_LN>(a, M, phases, st);
    case PRE_LN_MISH:
      return wgemm_kernel<C, PRE_LN_MISH>(a, M, phases, st);
    default:
      return wgemm_kernel<C, PRE_LN_MISH_ADD>(a, M, phases, st);
  }
}
// cfg 0..2: mma.sync configs (3xTF32 path); 10..12: wgmma configs (TF32 path)
cudaError_t gemm_dispatch(const GemmArgs* a, int cfg, int prec, int pre, int M, int phases, cudaStream_t st) {
  if (cfg == 10) return wgemm_pre<WCfg0>(a, pre, M, phases, st);
  if (cfg == 11) return wgemm_pre<WCfg1>(a, pre, M, phases, st);
  if (cfg == 12) return wgemm_pre<WCfg2>(a, pre, M, phases, st);
  if (prec == 3) {
    if (cfg == 0) return gemm_pre<0, 3>(a, pre, M, phases, st);
    if (cfg == 1) return gemm_pre<1, 3>(a, pre, M, phases, st);
    return gemm_pre<2, 3>(a, pre, M, phases, st);
  }
  if (cfg == 0) return gemm_pre<0, 1>(a, pre, M, phases, st);
  if (cfg == 1) return gemm_pre<1, 1>(a, pre, M, phases, st);
  return gemm_pre<2, 1>(a, pre, M, phases, st);
}
int set_attrs() {
  for (int cfg : {0, 1, 2, 10, 11, 12})
    for (int prec : {1, 3})
      for (int pre = 0; pre <= PRE_LN_MISH_ADD; ++pre) CK(gemm_dispatch(nullptr, cfg, prec, pre, 0, 0, 0));
  CK(cudaFuncSetAttribute(k_attn<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, ASmem<false>::bytes));
  CK(cudaFuncSetAttribute(k_attn<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, ASmem<true>::bytes));
  return 0;
}

// Fluent builder for GemmArgs.
struct GB {
  GemmArgs a{};
  explicit GB(const int* lens) {
    a.lens = lens;
    a.stride = 1;
    a.dil = 1;
    a.oscale = 1.f;
    a.ostride = 1;
    a.shift = 0;
  }
  GB& rows(int tcap, Len l) {
    a.tcap = tcap;
    a.rlen = l;
    return *this;
  }
  GB& in(const float* A, int lda, int cin, int tcap, Len l, int bmod = 0) {
    a.A = A;
    a.lda = lda;
    a.cin = cin;
    a.a_tcap = tcap;
    a.alen = l;
    a.a_bmod = bmod;
    return *this;
  }
  GB& conv(int taps, int dil, int pad, int stride = 1, int shift = 0) {
    a.K = taps * a.cin;
    a.dil = dil;
    a.pad = pad;
    a.stride = stride;
    a.shift = shift;
    return *this;
  }
  GB& ln(const float* g, const float* b, float eps, int kind = PRE_LN, const float* t = nullptr) {
    a.pre = kind;
    a.pg = g;
    a.pb = b;
    a.pt = t;
    a.eps = eps;
    return *this;
  }
  GB& snake(const float* al) {
    a.pre = PRE_SNAKE;
    a.pg = al;
    return *this;
  }
  GB& lrelu(float s) {
    a.pre = PRE_LRELU;
    a.slope = s;
    return *this;
  }
  GB& out(float* o, int ldo, int tcap, Len l, int ostride = 1, int ooff = 0, int o_lo = 0) {
    a.out = o;
    a.ldo = ldo;
    a.o_tcap = tcap;
    a.olen = l;
    a.ostride = ostride;
    a.ooff = ooff;
    a.o_lo = o_lo;
    return *this;
  }
  GB& bias(const float* b) {
    a.bias = b;
    return *this;
  }
  GB& act(int k, float s = 0.f) {
    a.act = k;
    a.aslope = s;
    return *this;
  }
  GB& aux(const float* p, int ld) {
    a.aux = p;
    a.ldaux = ld;
    return *this;
  }
  GB& aux2(const float* p, int ld) {
    a.aux2 = p;
    a.ldaux2 = ld;
    return *this;
  }
  GB& scale(float s) {
    a.oscale = s;
    return *this;
  }
};

int gemm(S3* s, const GB& gb, const GW& w, int nb, cudaStream_t st, int prec = 0) {
  GemmArgs a = gb.a;
  if (a.K == 0) a.K = a.cin;  // linear
  if (a.K != w.K || a.cin % 4 || a.lda % 4 || a.tcap % 128 || !a.out) {
    fprintf(stderr, "plow_s3gen: bad gemm (K %d vs %d, cin %d, lda %d, tcap %d)\n", a.K, w.K, a.cin, a.lda, a.tcap);
    return -1;
  }
  if (a.pre >= PRE_LN && (a.cin % 32 || a.cin > 512 || a.stride != 1 || a.dil != 1 || a.K / a.cin > 9 || a.shift)) {
    fprintf(stderr, "plow_s3gen: bad LN gemm\n");
    return -1;
  }
  if (a.o_tcap == 0) a.o_tcap = a.tcap;
  a.W = w.w;
  a.kpad = w.kpad;
  a.N = w.N;
  a.wz = (long)w.npad * w.kpad;
  const int M = nb * a.tcap, phases = w.phases;
  if (!prec) prec = s->prec;
  int cfg;
  if (prec == 3) {
    const long tl = (long)(M / 128) * cdiv(a.N, 128) * phases, tm = (long)(M / 128) * cdiv(a.N, 64) * phases;
    cfg = tl >= 120 ? 0 : tm >= 120 ? 1 : 2;
  } else {
    a.W = w.wr;
    const long t0 = (long)(M / 128) * cdiv(a.N, 128) * phases, t1 = (long)(M / 64) * cdiv(a.N, 128) * phases;
    cfg = t0 >= 132 ? 10 : t1 >= 100 ? 11 : 12;
  }
  cudaError_t e = gemm_dispatch(&a, cfg, prec, a.pre, M, phases, st);
  CK(e);
  ++s->nlaunch;
  return 0;
}

int attn(S3* s, bool rel, const ELW* el, const float* qkv, float* out, int nb, int tcap, Len l, cudaStream_t st) {
  AttnArgs a{};
  a.lens = s->d_lens;
  a.len = l;
  a.qkv = qkv;
  a.ldq = 1536;
  a.out = out;
  a.ldo = 512;
  a.tcap = tcap;
  if (rel) {
    a.prel = el->prel;
    a.rm = s->rm;
    a.pu = el->pu;
    a.pv = el->pv;
    CK(launch(k_attn<true>, dim3(tcap / 64, 8, nb), dim3(128), ASmem<true>::bytes, st, a));
  } else {
    CK(launch(k_attn<false>, dim3(tcap / 64, 8, nb), dim3(128), ASmem<false>::bytes, st, a));
  }
  ++s->nlaunch;
  return 0;
}

// ---- pipeline ----
int conformer(S3* s, const ELW& L, float* X, int nb, int tcap, Len len, cudaStream_t st) {
  Buf& u = s->buf;
  const int* lens = s->d_lens;
  RC(gemm(s, GB(lens).rows(tcap, len).in(X, 512, 512, tcap, len).ln(L.lnmg, L.lnmb, 1e-12f).bias(L.qkvb).out(u.qkv, 1536, tcap, len), L.qkv, nb, st));
  RC(attn(s, true, &L, u.qkv, u.att, nb, tcap, len, st));
  RC(gemm(s, GB(lens).rows(tcap, len).in(u.att, 512, 512, tcap, len).bias(L.outb).aux(X, 512).out(X, 512, tcap, len), L.out, nb, st));
  RC(gemm(s, GB(lens).rows(tcap, len).in(X, 512, 512, tcap, len).ln(L.lnfg, L.lnfb, 1e-12f).bias(L.ff1b).act(ACT_SILU).out(u.ff, 2048, tcap, len), L.ff1, nb, st));
  RC(gemm(s, GB(lens).rows(tcap, len).in(u.ff, 2048, 2048, tcap, len).bias(L.ff2b).aux(X, 512).out(X, 512, tcap, len), L.ff2, nb, st));
  return 0;
}

int ln_rows(S3* s, const float* x, int ldx, float* y, int ldy, const float* g, const float* b, float eps, float scale,
            int C, int nb, int tcap, Len len, cudaStream_t st) {
  const int nrows = nb * tcap;
  CK(launch(k_ln_rows, dim3(cdiv(nrows, 8)), dim3(256), 0, st, x, ldx, y, ldy, g, b, eps, scale, C, tcap,
            (const int*)s->d_lens, len, nrows));
  ++s->nlaunch;
  return 0;
}

// CFM transformer block on X (ld ldx); the FF2 result goes to `dst` (ld ldd; may be X).
int tblock(S3* s, const TBW& T, float* X, int ldx, float* dst, int ldd, int nb, int tcap, Len len, cudaStream_t st) {
  Buf& u = s->buf;
  const int* lens = s->d_lens;
  RC(gemm(s, GB(lens).rows(tcap, len).in(X, ldx, 256, tcap, len).ln(T.ln1g, T.ln1b, 1e-5f).out(u.qkv, 1536, tcap, len), T.qkv, nb, st));
  RC(attn(s, false, nullptr, u.qkv, u.att, nb, tcap, len, st));
  RC(gemm(s, GB(lens).rows(tcap, len).in(u.att, 512, 512, tcap, len).bias(T.outb).aux(X, ldx).out(X, ldx, tcap, len), T.out, nb, st));
  RC(gemm(s, GB(lens).rows(tcap, len).in(X, ldx, 256, tcap, len).ln(T.ln3g, T.ln3b, 1e-5f).bias(T.ff1b).act(ACT_GELU).out(u.ff, 1024, tcap, len), T.ff1, nb, st));
  RC(gemm(s, GB(lens).rows(tcap, len).in(u.ff, 1024, 1024, tcap, len).bias(T.ff2b).aux(X, ldx).out(dst, ldd, tcap, len), T.ff2, nb, st));
  return 0;
}

// CFM ResNet block j (1..13) on input X (cin, ld ldx) -> u.h
int resnet(S3* s, int j, int step, const float* X, int ldx, int cin, int nb, int tcap, Len len, cudaStream_t st) {
  Buf& u = s->buf;
  const RNW& R = s->rn[j];
  const int* lens = s->d_lens;
  const float* tv = s->tvec + ((long)step * 14 + j) * 256;
  RC(gemm(s, GB(lens).rows(tcap, len).in(X, ldx, cin, tcap, len).conv(3, 1, 2).bias(R.c1b).out(u.y1, 256, tcap, len), R.c1, nb, st));
  RC(gemm(s, GB(lens).rows(tcap, len).in(u.y1, 256, 256, tcap, len).conv(3, 1, 2).ln(R.ln1g, R.ln1b, 1e-5f, PRE_LN_MISH_ADD, tv).bias(R.c2b).out(u.y2, 256, tcap, len), R.c2, nb, st));
  RC(gemm(s, GB(lens).rows(tcap, len).in(X, ldx, cin, tcap, len).bias(R.rb).out(u.r, 256, tcap, len), R.r, nb, st));
  const int nrows = nb * tcap;
  CK(launch(k_resout, dim3(cdiv(nrows, 8)), dim3(256), 0, st, (const float*)u.y2, (const float*)u.r, R.ln2g, R.ln2b,
            u.h, 256, tcap, (const int*)lens, len, nrows));
  ++s->nlaunch;
  return 0;
}

// HiFT ResBlock: x -> out.  acc mode: out = (resblock(x)) / 3 + (acc_in ? out : 0)
int hift_resblock(S3* s, const RBW& R, const float* x, float* out, int C, int nb, int tcap, Len len, bool acc_mode,
                  bool acc_add, cudaStream_t st) {
  Buf& u = s->buf;
  const int* lens = s->d_lens;
  float* T = u.hb[3];
  float* ping[2] = {u.hb[4], u.hb[5]};
  const float* cur = x;
  const int dils[3] = {1, 3, 5};
  for (int d = 0; d < 3; ++d) {
    const int dl = dils[d];
    RC(gemm(s, GB(lens).rows(tcap, len).in(cur, C, C, tcap, len).conv(R.k, dl, dl * (R.k - 1) / 2).snake(R.a1[d]).bias(R.c1b[d]).out(T, C, tcap, len), R.c1[d], nb, st));
    GB g2 = GB(lens).rows(tcap, len).in(T, C, C, tcap, len).conv(R.k, 1, (R.k - 1) / 2).snake(R.a2[d]).bias(R.c2b[d]).aux(cur, C);
    if (d < 2) {
      RC(gemm(s, g2.out(ping[d], C, tcap, len), R.c2[d], nb, st));
      cur = ping[d];
    } else if (acc_mode) {
      g2.scale(1.0f / 3.0f);
      if (acc_add) g2.aux2(out, C);
      RC(gemm(s, g2.out(out, C, tcap, len), R.c2[d], nb, st));
    } else {
      RC(gemm(s, g2.out(out, C, tcap, len), R.c2[d], nb, st));
    }
  }
  return 0;
}

int enqueue(S3* s, const Caps& c, cudaStream_t st) {
  Buf& u = s->buf;
  const int* lens = s->d_lens;
  const int B = c.B, B2 = 2 * B;
  const Len LT{SP_TOK, 1, 0}, LT2{SP_TOK, 2, 0}, LM{SP_MEL, 1, 0}, LG{SP_GEN, 1, 0};
  s->nlaunch = 0;
  CK(launch(k_setup, dim3(1), dim3(kMaxB), 0, st, (const CallArgs*)s->d_call, s->d_lens));
  ++s->nlaunch;
  // ================= encoder =================
  CK(launch(k_embed, dim3(c.t0, B), dim3(kDE / 4), 0, st, (const CallArgs*)s->d_call, lens, (const int*)s->d_tok, s->emb, u.e0, c.t0));
  ++s->nlaunch;
  RC(gemm(s, GB(lens).rows(c.t0, LT).in(u.e0, 512, 512, c.t0, LT).bias(s->emb_w_b).out(u.e1, 512, c.t0, LT), s->emb_w, B, st));
  const float xscale = (float)std::sqrt(512.0);
  RC(ln_rows(s, u.e1, 512, u.e0, 512, s->emb_lng, s->emb_lnb, 1e-5f, xscale, 512, B, c.t0, LT, st));
  RC(gemm(s, GB(lens).rows(c.t0, LT).in(u.e0, 512, 512, c.t0, LT).conv(4, 1, 0).bias(s->la1b).act(ACT_LRELU, 0.01f).out(u.e1, 512, c.t0, LT), s->la1, B, st));
  RC(gemm(s, GB(lens).rows(c.t0, LT).in(u.e1, 512, 512, c.t0, LT).conv(3, 1, 2).bias(s->la2b).aux(u.e0, 512).out(u.e2, 512, c.t0, LT), s->la2, B, st));
  for (int i = 0; i < 6; ++i) RC(conformer(s, s->el[i], u.e2, B, c.t0, LT, st));
  RC(gemm(s, GB(lens).rows(c.t1, LM).in(u.e2, 512, 512, c.t0, LT2).conv(5, 1, 4, 1, 1).bias(s->upb).out(u.e1, 512, c.t1, LM), s->up, B, st));
  RC(gemm(s, GB(lens).rows(c.t1, LM).in(u.e1, 512, 512, c.t1, LM).bias(s->upe_b).out(u.e0, 512, c.t1, LM), s->upe_w, B, st));
  RC(ln_rows(s, u.e0, 512, u.e1, 512, s->upe_lng, s->upe_lnb, 1e-5f, xscale, 512, B, c.t1, LM, st));
  for (int i = 6; i < 10; ++i) RC(conformer(s, s->el[i], u.e1, B, c.t1, LM, st));
  RC(gemm(s, GB(lens).rows(c.t1, LM).in(u.e1, 512, 512, c.t1, LM).ln(s->aln_g, s->aln_b, 1e-5f).bias(s->projb).out(u.mu, kMel, c.t1, LM), s->proj, B, st));
  // ================= CFM =================
  CK(launch(k_cfm_init, dim3(c.t1, B), dim3(128), 0, st, (const CallArgs*)s->d_call, lens, (const float*)u.mu, u.x, u.z, u.ctx, c.t1));
  ++s->nlaunch;
  const RNW& R0 = s->rn[0];
  RC(gemm(s, GB(lens).rows(c.t1, LM).in(u.ctx, 240, 240, c.t1, LM).conv(3, 1, 2).bias(R0.c1b).out(u.E1, 256, c.t1, LM), R0.c1c, B2, st));
  RC(gemm(s, GB(lens).rows(c.t1, LM).in(u.ctx, 240, 240, c.t1, LM).bias(R0.rb).out(u.R1, 256, c.t1, LM), R0.rc, B2, st));
  for (int k = 0; k < kSteps; ++k) {
    // down ResNet (x channels only; the context part is E1 / R1)
    RC(gemm(s, GB(lens).rows(c.t1, LM).in(u.x, kMel, kMel, c.t1, LM, B).conv(3, 1, 2).aux(u.E1, 256).out(u.y1, 256, c.t1, LM), R0.c1, B2, st));
    RC(gemm(s, GB(lens).rows(c.t1, LM).in(u.y1, 256, 256, c.t1, LM).conv(3, 1, 2).ln(R0.ln1g, R0.ln1b, 1e-5f, PRE_LN_MISH_ADD, s->tvec + (long)k * 14 * 256).bias(R0.c2b).out(u.y2, 256, c.t1, LM), R0.c2, B2, st));
    RC(gemm(s, GB(lens).rows(c.t1, LM).in(u.x, kMel, kMel, c.t1, LM, B).aux(u.R1, 256).out(u.r, 256, c.t1, LM), R0.r, B2, st));
    {
      const int nrows = B2 * c.t1;
      CK(launch(k_resout, dim3(cdiv(nrows, 8)), dim3(256), 0, st, (const float*)u.y2, (const float*)u.r, R0.ln2g, R0.ln2b, u.h, 256, c.t1, lens, LM, nrows));
      ++s->nlaunch;
    }
    for (int i = 0; i < 4; ++i) {
      const bool last = i == 3;
      RC(tblock(s, s->tb[i], u.h, 256, last ? u.cat + 256 : u.h, last ? 512 : 256, B2, c.t1, LM, st));
    }
    RC(gemm(s, GB(lens).rows(c.t1, LM).in(u.cat + 256, 512, 256, c.t1, LM).conv(3, 1, 2).bias(s->downb).out(u.h, 256, c.t1, LM), s->downc, B2, st));
    for (int j = 1; j <= 12; ++j) {
      RC(resnet(s, j, k, u.h, 256, 256, B2, c.t1, LM, st));
      for (int i = 0; i < 4; ++i) {
        const bool last = j == 12 && i == 3;
        RC(tblock(s, s->tb[4 * j + i], u.h, 256, last ? u.cat : u.h, last ? 512 : 256, B2, c.t1, LM, st));
      }
    }
    RC(resnet(s, 13, k, u.cat, 512, 512, B2, c.t1, LM, st));
    for (int i = 0; i < 4; ++i) RC(tblock(s, s->tb[52 + i], u.h, 256, u.h, 256, B2, c.t1, LM, st));
    RC(gemm(s, GB(lens).rows(c.t1, LM).in(u.h, 256, 256, c.t1, LM).conv(3, 1, 2).bias(s->upcb).out(u.y1, 256, c.t1, LM), s->upc, B2, st));
    RC(gemm(s, GB(lens).rows(c.t1, LM).in(u.y1, 256, 256, c.t1, LM).conv(3, 1, 2).bias(s->finb).out(u.y2, 256, c.t1, LM), s->fin, B2, st));
    RC(gemm(s, GB(lens).rows(c.t1, LM).in(u.y2, 256, 256, c.t1, LM).ln(s->fin_lng, s->fin_lnb, 1e-5f, PRE_LN_MISH).bias(s->cprojb).out(u.d, kMel, c.t1, LM), s->cproj, B2, st));
    const float dt = s->tspan[k + 1] - s->tspan[k];
    CK(launch(k_euler, dim3(c.t1, B), dim3(kMel), 0, st, (const CallArgs*)s->d_call, lens, u.x, (const float*)u.d, c.t1, dt));
    ++s->nlaunch;
  }
  // ================= HiFT =================
  CK(launch(k_melgen, dim3(c.g, B), dim3(kMel), 0, st, (const CallArgs*)s->d_call, lens, (const float*)u.x, c.t1, u.melg, c.g));
  ++s->nlaunch;
  {  // F0 predictor
    const float* in = u.melg;
    int cin = kMel;
    float* outs[2] = {u.fa, u.fb};
    for (int i = 0; i < 5; ++i) {
      float* o = outs[i & 1];
      RC(gemm(s, GB(lens).rows(c.g, LG).in(in, cin, cin, c.g, LG).conv(3, 1, 1).bias(s->f0cb[i]).act(ACT_ELU).out(o, 512, c.g, LG), s->f0c[i], B, st));
      in = o;
      cin = 512;
    }
    RC(gemm(s, GB(lens).rows(c.g, LG).in(in, 512, 512, c.g, LG).bias(s->f0clsb).act(ACT_ABS).out(u.f0, 1, c.g, LG), s->f0cls, B, st));
  }
  CK(launch(k_f0scan, dim3(B), dim3(256), 0, st, lens, (const float*)u.f0, c.g, u.f0p));
  ++s->nlaunch;
  CK(launch(k_source, dim3(cdiv(c.wav, 256), B), dim3(256), 0, st, (const CallArgs*)s->d_call, lens, (const float*)u.f0, (const double*)u.f0p, c.g, s->srcw, s->srcb, u.src, c.wav, u.dbg_noise, u.dbg_phase));
  ++s->nlaunch;
  CK(launch(k_stft, dim3(cdiv(c.h2, 128), B), dim3(128), 0, st, lens, (const float*)u.src, c.wav, u.stft, c.h2));
  ++s->nlaunch;
  float* P = u.hb[0];
  float* ACC = u.hb[6];
  float* X = u.hb[1];
  float* SI = u.hb[2];
  RC(gemm(s, GB(lens).rows(c.g, LG).in(u.melg, kMel, kMel, c.g, LG).conv(7, 1, 3).bias(s->hpreb).out(P, 512, c.g, LG), s->hpre, B, st));
  {
    const int cins[3] = {512, 256, 128}, us[3] = {8, 5, 3}, ks[3] = {16, 11, 7};
    const int incap[3] = {c.g, c.h0, c.h1}, qcap[3] = {c.q0, c.q1, c.q2}, ocap[3] = {c.h0, c.h1, c.h2};
    const Len lin[3] = {{SP_GEN, 1, 0}, {SP_H0, 1, 0}, {SP_H1, 1, 0}};
    const Len lq[3] = {{SP_GEN, 1, 1}, {SP_H0, 1, 1}, {SP_H1, 1, 1}};
    const Len lo[3] = {{SP_H0, 1, 0}, {SP_H1, 1, 0}, {SP_H2, 1, 0}};
    const int sdk[3] = {30, 6, 1}, sds[3] = {15, 3, 1}, sdp[3] = {7, 1, 0};
    const int rbk[3] = {3, 7, 11};
    for (int i = 0; i < 3; ++i) {
      const int Co = cins[i] / 2;
      // source branch: si = source_resblock(source_down(stft))
      RC(gemm(s, GB(lens).rows(ocap[i], lo[i]).in(u.stft, 20, 20, c.h2, {SP_H2, 1, 0}).conv(sdk[i], 1, sdp[i], sds[i]).bias(s->hsdb[i]).out(X, Co, ocap[i], lo[i]), s->hsd[i], B, st));
      RC(hift_resblock(s, s->sr[i], X, SI, Co, B, ocap[i], lo[i], false, false, st));
      // x = ConvTranspose(lrelu(x, 0.1)) [+ reflect pad on the last] + si
      const int taps = (ks[i] + us[i] - 1) / us[i], pad = (ks[i] - us[i]) / 2;
      const bool last = i == 2;
      GB g = GB(lens).rows(qcap[i], lq[i]).in(P, cins[i], cins[i], incap[i], lin[i]).conv(taps, -1, 0).lrelu(0.1f).bias(s->hupb[i]).aux(SI, Co);
      g.out(X, Co, ocap[i], lo[i], us[i], -pad + (last ? 1 : 0), last ? 1 : 0);
      g.a.reflect = last ? 1 : 0;
      RC(gemm(s, g, s->hup[i], B, st));
      for (int j = 0; j < 3; ++j) {
        const RBW& R = s->rb[i * 3 + j];
        if (R.k != rbk[j]) return -5;
        RC(hift_resblock(s, R, X, ACC, Co, B, ocap[i], lo[i], true, j > 0, st));
      }
      std::swap(P, ACC);
    }
  }
  RC(gemm(s, GB(lens).rows(c.h2, {SP_H2, 1, 0}).in(P, 64, 64, c.h2, {SP_H2, 1, 0}).conv(7, 1, 3).lrelu(0.01f).bias(s->hpostb).out(u.post, 18, c.h2, {SP_H2, 1, 0}), s->hpost, B, st));
  CK(launch(k_istft, dim3(cdiv(c.wav, 256), B), dim3(256), 0, st, lens, (const float*)u.post, c.h2, u.wav, c.wav));
  ++s->nlaunch;
  return 0;
}

Caps make_caps(int B, int maxT0, int maxG) {
  Caps c{};
  c.B = B;
  c.t0 = rup(maxT0, 128);
  c.t1 = 2 * c.t0;
  c.g = rup(maxG, 128);
  c.h0 = 8 * c.g;
  c.h1 = 40 * c.g;
  c.h2 = 120 * c.g + 128;
  c.q0 = c.g + 128;
  c.q1 = c.h0 + 128;
  c.q2 = c.h1 + 128;
  c.wav = kUp * c.g;
  return c;
}

int build_graph(S3* s, const Caps& c, GraphEntry* ge) {
  cudaGraph_t g = nullptr;
  CK(cudaStreamBeginCapture(s->cap_stream, cudaStreamCaptureModeThreadLocal));
  int r = enqueue(s, c, s->cap_stream);
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
  ge->graph = g;
  ge->exec = ex;
  ge->launches = s->nlaunch;
  return 0;
}

// ---- weights ----
int read_file(const char* path, std::vector<char>& buf) {
  FILE* f = fopen(path, "rb");
  if (!f) return -4;
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  buf.resize(sz);
  bool ok = sz > 0 && fread(buf.data(), 1, sz, f) == (size_t)sz;
  fclose(f);
  return ok ? 0 : -4;
}

int parse_blob(const std::vector<char>& buf, const char* magic, std::map<std::string, S3::T>& out) {
  const long sz = (long)buf.size();
  if (sz < 16 || memcmp(buf.data(), magic, 8)) return -5;
  uint32_t ver, cnt;
  memcpy(&ver, &buf[8], 4);
  memcpy(&cnt, &buf[12], 4);
  if (ver != 1) return -5;
  long p = 16;
  for (uint32_t i = 0; i < cnt; ++i) {
    uint32_t nl, dt, nd;
    if (p + 4 > sz) return -5;
    memcpy(&nl, &buf[p], 4);
    p += 4;
    if (p + nl + 8 > sz) return -5;
    std::string name(&buf[p], nl);
    p += nl;
    memcpy(&dt, &buf[p], 4);
    memcpy(&nd, &buf[p + 4], 4);
    p += 8;
    if (nd > 8 || p + 4L * nd + 16 > sz) return -5;
    S3::T t;
    t.dtype = (int)dt;
    long numel = 1;
    for (uint32_t d = 0; d < nd; ++d) {
      uint32_t v;
      memcpy(&v, &buf[p + 4 * d], 4);
      t.dims.push_back((int)v);
      numel *= v;
    }
    p += 4L * nd;
    uint64_t off, nb;
    memcpy(&off, &buf[p], 8);
    memcpy(&nb, &buf[p + 8], 8);
    p += 16;
    if (off + nb > (uint64_t)sz || off % 16 || (long)nb != numel * 4) return -5;
    t.off = (long)off;
    t.nb = (long)nb;
    out[name] = t;
  }
  return 0;
}

struct Loader {
  S3* s;
  bool ok = true;
  std::string bad;
  const S3::T* find(const std::string& n, long numel) {
    auto it = s->tens.find(n);
    if (it == s->tens.end() || it->second.nb != numel * 4) {
      if (ok) bad = n;
      ok = false;
      return nullptr;
    }
    return &it->second;
  }
  const float* vec(const std::string& n, long numel) {
    const S3::T* t = find(n, numel);
    return t ? (const float*)((const char*)s->wdev + t->off) : nullptr;
  }
  const float* host(const std::string& n, long numel) {
    const S3::T* t = find(n, numel);
    return t ? (const float*)(s->blob.data() + t->off) : nullptr;
  }
  // GEMM weight [phases][N][K] -> padded device [phases][roundup(N,128)][roundup(K,32)]
  GW gw(const std::string& n, int N, int K, int phases = 1) {
    GW g;
    const float* src = host(n, (long)phases * N * K);
    if (!src) return g;
    g.N = N;
    g.K = K;
    g.phases = phases;
    g.npad = rup(N, 128);
    g.kpad = rup(K, 64);
    std::vector<float> h((size_t)phases * g.npad * g.kpad, 0.f);
    for (int z = 0; z < phases; ++z)
      for (int r = 0; r < N; ++r)
        memcpy(&h[((size_t)z * g.npad + r) * g.kpad], src + ((size_t)z * N + r) * K, (size_t)K * 4);
    float* d = nullptr;
    if (cudaMalloc(&d, h.size() * 4) != cudaSuccess || cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice) != cudaSuccess) {
      ok = false;
      bad = "cuda alloc " + n;
      return g;
    }
    s->allocs.push_back(d);
    g.w = d;
    for (float& v : h) {  // round to nearest (ties away), as cvt.rna.tf32.f32
      uint32_t u;
      memcpy(&u, &v, 4);
      if ((u & 0x7f800000u) != 0x7f800000u) u = (u + 0x1000u) & 0xffffe000u;
      memcpy(&v, &u, 4);
    }
    float* dr = nullptr;
    if (cudaMalloc(&dr, h.size() * 4) != cudaSuccess || cudaMemcpy(dr, h.data(), h.size() * 4, cudaMemcpyHostToDevice) != cudaSuccess) {
      ok = false;
      bad = "cuda alloc " + n;
      return g;
    }
    s->allocs.push_back(dr);
    g.wr = dr;
    return g;
  }
};

int load_weights(S3* s, const char* path) {
  RC(read_file(path, s->blob));
  RC(parse_blob(s->blob, "S3GENW01", s->tens));
  CK(cudaMalloc(&s->wdev, s->blob.size()));
  CK(cudaMemcpy(s->wdev, s->blob.data(), s->blob.size(), cudaMemcpyHostToDevice));
  Loader L{s};
  s->emb = L.vec("enc.emb", (long)kVocab * kDE);
  s->emb_w = L.gw("enc.embed.w", 512, 512);
  s->emb_w_b = L.vec("enc.embed.b", 512);
  s->emb_lng = L.vec("enc.embed.ln.g", 512);
  s->emb_lnb = L.vec("enc.embed.ln.b", 512);
  s->la1 = L.gw("enc.la1.w", 512, 4 * 512);
  s->la1b = L.vec("enc.la1.b", 512);
  s->la2 = L.gw("enc.la2.w", 512, 3 * 512);
  s->la2b = L.vec("enc.la2.b", 512);
  for (int i = 0; i < 10; ++i) {
    ELW& e = s->el[i];
    const std::string p = "enc.L" + std::to_string(i) + ".";
    e.lnmg = L.vec(p + "ln_mha.g", 512);
    e.lnmb = L.vec(p + "ln_mha.b", 512);
    e.qkv = L.gw(p + "qkv.w", 1536, 512);
    e.qkvb = L.vec(p + "qkv.b", 1536);
    e.pos = L.gw(p + "pos.w", 512, 512);
    e.pu = L.vec(p + "pos_u", 512);
    e.pv = L.vec(p + "pos_v", 512);
    e.out = L.gw(p + "out.w", 512, 512);
    e.outb = L.vec(p + "out.b", 512);
    e.lnfg = L.vec(p + "ln_ff.g", 512);
    e.lnfb = L.vec(p + "ln_ff.b", 512);
    e.ff1 = L.gw(p + "ff1.w", 2048, 512);
    e.ff1b = L.vec(p + "ff1.b", 2048);
    e.ff2 = L.gw(p + "ff2.w", 512, 2048);
    e.ff2b = L.vec(p + "ff2.b", 512);
  }
  s->up = L.gw("enc.up.w", 512, 5 * 512);
  s->upb = L.vec("enc.up.b", 512);
  s->upe_w = L.gw("enc.up_embed.w", 512, 512);
  s->upe_b = L.vec("enc.up_embed.b", 512);
  s->upe_lng = L.vec("enc.up_embed.ln.g", 512);
  s->upe_lnb = L.vec("enc.up_embed.ln.b", 512);
  s->aln_g = L.vec("enc.after_ln.g", 512);
  s->aln_b = L.vec("enc.after_ln.b", 512);
  s->proj = L.gw("enc.proj.w", 80, 512);
  s->projb = L.vec("enc.proj.b", 80);
  if (const float* w = L.host("spk.w", 80 * 192)) s->spk_w.assign(w, w + 80 * 192);
  if (const float* w = L.host("spk.b", 80)) s->spk_b.assign(w, w + 80);
  if (const float* t = L.host("cfm.t_span", kSteps + 1)) memcpy(s->tspan, t, sizeof(s->tspan));
  s->tvec = L.vec("cfm.tvec", (long)kSteps * 14 * 256);
  for (int j = 0; j < 14; ++j) {
    RNW& r = s->rn[j];
    const std::string p = "cfm.r" + std::to_string(j) + ".";
    const int cin = j == 13 ? 512 : 256;
    if (j == 0) {
      r.c1 = L.gw(p + "c1x.w", 256, 3 * 80);
      r.c1c = L.gw(p + "c1c.w", 256, 3 * 240);
      r.r = L.gw(p + "rx.w", 256, 80);
      r.rc = L.gw(p + "rc.w", 256, 240);
    } else {
      r.c1 = L.gw(p + "c1.w", 256, 3 * cin);
      r.r = L.gw(p + "r.w", 256, cin);
    }
    r.c1b = L.vec(p + "c1.b", 256);
    r.rb = L.vec(p + "r.b", 256);
    r.ln1g = L.vec(p + "ln1.g", 256);
    r.ln1b = L.vec(p + "ln1.b", 256);
    r.c2 = L.gw(p + "c2.w", 256, 3 * 256);
    r.c2b = L.vec(p + "c2.b", 256);
    r.ln2g = L.vec(p + "ln2.g", 256);
    r.ln2b = L.vec(p + "ln2.b", 256);
  }
  for (int k = 0; k < 56; ++k) {
    TBW& t = s->tb[k];
    const std::string p = "cfm.tb" + std::to_string(k) + ".";
    t.ln1g = L.vec(p + "ln1.g", 256);
    t.ln1b = L.vec(p + "ln1.b", 256);
    t.qkv = L.gw(p + "qkv.w", 1536, 256);
    t.out = L.gw(p + "out.w", 256, 512);
    t.outb = L.vec(p + "out.b", 256);
    t.ln3g = L.vec(p + "ln3.g", 256);
    t.ln3b = L.vec(p + "ln3.b", 256);
    t.ff1 = L.gw(p + "ff1.w", 1024, 256);
    t.ff1b = L.vec(p + "ff1.b", 1024);
    t.ff2 = L.gw(p + "ff2.w", 256, 1024);
    t.ff2b = L.vec(p + "ff2.b", 256);
  }
  s->downc = L.gw("cfm.down.w", 256, 768);
  s->downb = L.vec("cfm.down.b", 256);
  s->upc = L.gw("cfm.upc.w", 256, 768);
  s->upcb = L.vec("cfm.upc.b", 256);
  s->fin = L.gw("cfm.fin.w", 256, 768);
  s->finb = L.vec("cfm.fin.b", 256);
  s->fin_lng = L.vec("cfm.fin.ln.g", 256);
  s->fin_lnb = L.vec("cfm.fin.ln.b", 256);
  s->cproj = L.gw("cfm.proj.w", 80, 256);
  s->cprojb = L.vec("cfm.proj.b", 80);
  for (int i = 0; i < 5; ++i) {
    const std::string p = "hift.f0.c" + std::to_string(i) + ".";
    s->f0c[i] = L.gw(p + "w", 512, 3 * (i ? 512 : 80));
    s->f0cb[i] = L.vec(p + "b", 512);
  }
  s->f0cls = L.gw("hift.f0.cls.w", 1, 512);
  s->f0clsb = L.vec("hift.f0.cls.b", 1);
  s->srcw = L.vec("hift.src.w", 9);
  s->srcb = L.vec("hift.src.b", 1);
  s->hpre = L.gw("hift.pre.w", 512, 7 * 80);
  s->hpreb = L.vec("hift.pre.b", 512);
  {
    const int cins[3] = {512, 256, 128}, us[3] = {8, 5, 3}, ks[3] = {16, 11, 7};
    const int sdk[3] = {30, 6, 1}, rsk[3] = {7, 7, 11}, rbk[3] = {3, 7, 11};
    for (int i = 0; i < 3; ++i) {
      const int Co = cins[i] / 2, taps = (ks[i] + us[i] - 1) / us[i];
      const std::string p = "hift.up" + std::to_string(i) + ".";
      s->hup[i] = L.gw(p + "w", Co, taps * cins[i], us[i]);
      s->hupb[i] = L.vec(p + "b", Co);
      const std::string q = "hift.sd" + std::to_string(i) + ".";
      s->hsd[i] = L.gw(q + "w", Co, sdk[i] * 20);
      s->hsdb[i] = L.vec(q + "b", Co);
      auto rbl = [&](RBW& R, const std::string& pre, int k) {
        R.k = k;
        for (int d = 0; d < 3; ++d) {
          const std::string x = pre + ".d" + std::to_string(d) + ".";
          R.a1[d] = L.vec(x + "a1", Co);
          R.a2[d] = L.vec(x + "a2", Co);
          R.c1[d] = L.gw(x + "c1.w", Co, k * Co);
          R.c2[d] = L.gw(x + "c2.w", Co, k * Co);
          R.c1b[d] = L.vec(x + "c1.b", Co);
          R.c2b[d] = L.vec(x + "c2.b", Co);
        }
      };
      rbl(s->sr[i], "hift.sr" + std::to_string(i), rsk[i]);
      for (int j = 0; j < 3; ++j) rbl(s->rb[i * 3 + j], "hift.rb" + std::to_string(i * 3 + j), rbk[j]);
    }
  }
  s->hpost = L.gw("hift.post.w", 18, 7 * 64);
  s->hpostb = L.vec("hift.post.b", 18);
  const float* win = L.host("hift.window", 16);
  const S3::T* pe = L.find("enc.pe", (long)kPeRows * 512);
  if (!L.ok) {
    fprintf(stderr, "plow_s3gen: missing or mis-sized tensor %s\n", L.bad.c_str());
    return -5;
  }
  // constants: DFT tables, window, fade
  float cs[16], sn[16], fade[480];
  for (int n = 0; n < 16; ++n) {
    cs[n] = (float)std::cos(2.0 * M_PI * n / 16.0);
    sn[n] = (float)std::sin(2.0 * M_PI * n / 16.0);
  }
  for (int i = 0; i < 480; ++i) {
    const float x = (float)(M_PI - M_PI * i / 479.0);
    fade[i] = (std::cos(x) + 1.0f) / 2.0f;
  }
  CK(cudaMemcpyToSymbol(c_cos16, cs, sizeof(cs)));
  CK(cudaMemcpyToSymbol(c_sin16, sn, sizeof(sn)));
  CK(cudaMemcpyToSymbol(c_win, win, 16 * 4));
  CK(cudaMemcpyToSymbol(c_fade, fade, sizeof(fade)));
  // Relative-position tables: prel[r + rm - 1] = linear_pos(pe(r)), r in [-(rm-1), rm-1].
  {
    const int R = 2 * s->rm - 1, Rcap = rup(R, 128);
    if (s->rm - 1 > kPeCenter) return -6;
    std::vector<float> h((size_t)Rcap * 512, 0.f);
    const float* pet = (const float*)(s->blob.data() + pe->off);
    for (int i = 0; i < R; ++i) {
      const int r = i - (s->rm - 1);
      memcpy(&h[(size_t)i * 512], pet + (size_t)(kPeCenter - r) * 512, 512 * 4);
    }
    float* dpe = nullptr;
    int* dl = nullptr;
    CK(cudaMalloc(&dpe, h.size() * 4));
    CK(cudaMemcpy(dpe, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dl, NSP * kLS * 4));
    std::vector<int> hl(NSP * kLS, 0);
    hl[SP_AUX * kLS] = R;
    CK(cudaMemcpy(dl, hl.data(), hl.size() * 4, cudaMemcpyHostToDevice));
    const Len la{SP_AUX, 1, 0};
    for (int i = 0; i < 10; ++i) {
      float* d = nullptr;
      CK(cudaMalloc(&d, (size_t)Rcap * 512 * 4));
      s->allocs.push_back(d);
      s->el[i].prel = d;
      RC(gemm(s, GB(dl).rows(Rcap, la).in(dpe, 512, 512, Rcap, la).out(d, 512, Rcap, la), s->el[i].pos, 1, 0, 3));
    }
    CK(cudaDeviceSynchronize());
    cudaFree(dpe);
    cudaFree(dl);
  }
  return 0;
}

int alloc_buffers(S3* s) {
  const long B = s->max_batch, T1 = 2L * s->t0max, G = s->gmax;
  const long HB = std::max({8 * G * 256, 40 * G * 128, (120 * G + 128) * 64});
  const long H2 = 120 * G + 128, W = kUp * G;
  Buf& u = s->buf;
  struct A {
    float** p;
    long n;
  } list[] = {
      {&u.e0, B * T1 * 512}, {&u.e1, B * T1 * 512}, {&u.e2, B * T1 * 512}, {&u.qkv, 2 * B * T1 * 1536},
      {&u.att, 2 * B * T1 * 512}, {&u.ff, 2 * B * T1 * 1024}, {&u.mu, B * T1 * 80}, {&u.x, B * T1 * 80},
      {&u.z, B * T1 * 80}, {&u.ctx, 2 * B * T1 * 240}, {&u.E1, 2 * B * T1 * 256}, {&u.R1, 2 * B * T1 * 256},
      {&u.y1, 2 * B * T1 * 256}, {&u.y2, 2 * B * T1 * 256}, {&u.r, 2 * B * T1 * 256}, {&u.h, 2 * B * T1 * 256},
      {&u.cat, 2 * B * T1 * 512}, {&u.d, 2 * B * T1 * 80}, {&u.melg, B * G * 80}, {&u.fa, B * G * 512},
      {&u.fb, B * G * 512}, {&u.f0, B * G}, {&u.src, B * W}, {&u.stft, B * H2 * 20}, {&u.post, B * H2 * 18},
      {&u.wav, B * W},
  };
  for (auto& a : list) {
    CK(cudaMalloc(a.p, a.n * 4));
    CK(cudaMemset(*a.p, 0, a.n * 4));
    s->allocs.push_back(*a.p);
  }
  for (int i = 0; i < 7; ++i) {
    CK(cudaMalloc(&u.hb[i], B * HB * 4));
    CK(cudaMemset(u.hb[i], 0, B * HB * 4));
    s->allocs.push_back(u.hb[i]);
  }
  CK(cudaMalloc(&u.f0p, B * (G + 1) * 8));
  s->allocs.push_back(u.f0p);
  if (s->debug) {
    CK(cudaMalloc(&u.dbg_noise, B * 9 * W * 4));
    CK(cudaMalloc(&u.dbg_phase, B * 9 * 4));
    s->allocs.push_back(u.dbg_noise);
    s->allocs.push_back(u.dbg_phase);
  }
  CK(cudaMalloc(&s->d_lens, NSP * kLS * 4));
  CK(cudaMemset(s->d_lens, 0, NSP * kLS * 4));
  CK(cudaMalloc(&s->d_call, sizeof(CallArgs)));
  CK(cudaMallocHost(&s->h_call, sizeof(CallArgs)));
  const long ntok = B * (s->max_tokens + 16);
  CK(cudaMalloc(&s->d_tok, ntok * 4));
  CK(cudaMallocHost(&s->h_tok, ntok * 4));
  CK(cudaMallocHost(&s->h_wav, B * W * 4));
  s->allocs.push_back(s->d_lens);
  s->allocs.push_back(s->d_call);
  s->allocs.push_back(s->d_tok);
  return 0;
}

}  // namespace

extern "C" int plow_s3gen_destroy(void* hp) {
  if (!hp) return -1;
  S3* s = (S3*)hp;
  cudaSetDevice(s->device);
  cudaDeviceSynchronize();
  for (auto& kv : s->graphs) {
    if (kv.second.exec) cudaGraphExecDestroy(kv.second.exec);
    if (kv.second.graph) cudaGraphDestroy(kv.second.graph);
  }
  for (auto& v : s->voices) {
    cudaFree(v.ptok);
    cudaFree(v.pfeat);
    cudaFree(v.spks);
  }
  for (void* p : s->allocs) cudaFree(p);
  if (s->h_call) cudaFreeHost(s->h_call);
  if (s->h_tok) cudaFreeHost(s->h_tok);
  if (s->h_wav) cudaFreeHost(s->h_wav);
  if (s->stream) cudaStreamDestroy(s->stream);
  if (s->cap_stream) cudaStreamDestroy(s->cap_stream);
  cudaFree(s->wdev);
  delete s;
  return 0;
}

extern "C" int plow_s3gen_create(int device, const char* weights_path, int max_batch, int max_tokens, void** out) {
  if (!out || !weights_path || max_batch < 1 || max_batch > kMaxB || max_tokens < 1) return -1;
  *out = nullptr;
  S3* s = new S3();
  s->device = device;
  s->max_batch = max_batch;
  s->max_tokens = max_tokens;
  s->max_prompt = 320;
  if (const char* e = getenv("PLOW_S3GEN_MAX_PROMPT")) s->max_prompt = std::max(1, atoi(e));
  if (const char* e = getenv("PLOW_S3GEN_GRAPH")) s->use_graph = e[0] != '0';
  if (const char* e = getenv("PLOW_S3GEN_PREC")) s->prec = strcmp(e, "3xtf32") == 0 ? 3 : 1;
  if (const char* e = getenv("PLOW_S3GEN_PDL")) g_pdl = e[0] != '0';
  if (const char* e = getenv("PLOW_S3GEN_DEBUG")) s->debug = e[0] == '1';
  auto fail = [&](int rc) {
    plow_s3gen_destroy(s);
    return rc;
  };
  if (cudaSetDevice(device) != cudaSuccess) return fail(-2);
  s->t0max = rup(max_tokens + s->max_prompt, 128);
  s->gmax = rup(2 * max_tokens + 16, 128);
  s->rm = 2 * s->t0max + 128;
  if (int r = set_attrs()) return fail(r);
  if (int r = load_weights(s, weights_path)) return fail(r);
  if (int r = alloc_buffers(s)) return fail(r);
  if (cudaStreamCreateWithFlags(&s->stream, cudaStreamNonBlocking) != cudaSuccess ||
      cudaStreamCreateWithFlags(&s->cap_stream, cudaStreamNonBlocking) != cudaSuccess)
    return fail(-2);
  if (cudaDeviceSynchronize() != cudaSuccess) return fail(-2);
  *out = s;
  return 0;
}

extern "C" int plow_s3gen_add_voice(void* hp, const char* voice_path, int* voice_id) {
  if (!hp || !voice_path || !voice_id) return -1;
  S3* s = (S3*)hp;
  std::vector<char> buf;
  RC(read_file(voice_path, buf));
  std::map<std::string, S3::T> t;
  RC(parse_blob(buf, "S3GENV01", t));
  auto pt = t.find("prompt_token"), pf = t.find("prompt_feat"), em = t.find("embedding");
  if (pt == t.end() || pf == t.end() || em == t.end() || pt->second.dtype != 1 || pf->second.dims.size() != 2 ||
      pf->second.dims[1] != kMel || em->second.nb != 192 * 4)
    return -5;
  Voice v;
  v.P = (int)(pt->second.nb / 4);
  v.Pf = pf->second.dims[0];
  if (v.P < 1 || v.P > s->max_prompt || v.Pf > 2 * v.P + 1 || v.Pf < 2 * v.P - 16) return -6;
  const int* toks = (const int*)(buf.data() + pt->second.off);
  for (int i = 0; i < v.P; ++i)
    if (toks[i] < 0 || toks[i] >= kVocab) return -5;
  // spks = spk_affine(normalize(embedding))  (F.normalize: x / max(||x||, 1e-12))
  const float* e = (const float*)(buf.data() + em->second.off);
  double nrm = 0.0;
  for (int i = 0; i < 192; ++i) nrm += (double)e[i] * e[i];
  nrm = std::max(std::sqrt(nrm), 1e-12);
  float spks[kMel];
  for (int o = 0; o < kMel; ++o) {
    double acc = s->spk_b[o];
    for (int i = 0; i < 192; ++i) acc += (double)s->spk_w[o * 192 + i] * ((double)e[i] / nrm);
    spks[o] = (float)acc;
  }
  CK(cudaSetDevice(s->device));
  CK(cudaMalloc(&v.ptok, v.P * 4));
  CK(cudaMalloc(&v.pfeat, (size_t)v.Pf * kMel * 4));
  CK(cudaMalloc(&v.spks, kMel * 4));
  CK(cudaMemcpy(v.ptok, toks, v.P * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(v.pfeat, buf.data() + pf->second.off, (size_t)v.Pf * kMel * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(v.spks, spks, kMel * 4, cudaMemcpyHostToDevice));
  s->voices.push_back(v);
  *voice_id = (int)s->voices.size() - 1;
  return 0;
}

extern "C" int plow_s3gen_synthesize_batch_host(void* hp, int B, const int* voice_ids, const int32_t* tokens,
                                                const int* n_tokens, float* wav_out, int max_samples,
                                                const unsigned long long* seeds, int* n_samples_out) {
  if (!hp || B < 1 || !voice_ids || !tokens || !n_tokens || !wav_out || !seeds || !n_samples_out) return -1;
  S3* s = (S3*)hp;
  if (B > s->max_batch) return -6;
  CallArgs& ca = *s->h_call;
  memset(&ca, 0, sizeof(ca));
  ca.B = B;
  int maxT0 = 0, maxG = 0, off = 0;
  bool small = false;
  for (int b = 0; b < B; ++b) {
    const int vid = voice_ids[b], n = n_tokens[b];
    if (vid < 0 || vid >= (int)s->voices.size() || n < 1) return -1;
    if (n > s->max_tokens) return -6;
    const Voice& v = s->voices[vid];
    const int T0 = v.P + n, G = 2 * T0 - v.Pf;
    if (G < 1) return -1;
    n_samples_out[b] = kUp * G;
    small |= kUp * G > max_samples;
    maxT0 = std::max(maxT0, T0);
    maxG = std::max(maxG, G);
    ca.n_tok[b] = n;
    ca.P[b] = v.P;
    ca.Pf[b] = v.Pf;
    ca.tok_off[b] = off;
    ca.ptok[b] = v.ptok;
    ca.pfeat[b] = v.pfeat;
    ca.spks[b] = v.spks;
    ca.seed[b] = seeds[b];
    off += n;
  }
  if (small) return -7;
  if (rup(maxT0, 128) > s->t0max || rup(maxG, 128) > s->gmax) return -6;
  const Caps c = make_caps(B, maxT0, maxG);
  CK(cudaSetDevice(s->device));
  memcpy(s->h_tok, tokens, (size_t)off * 4);
  CK(cudaMemcpyAsync(s->d_call, s->h_call, sizeof(CallArgs), cudaMemcpyHostToDevice, s->stream));
  CK(cudaMemcpyAsync(s->d_tok, s->h_tok, (size_t)off * 4, cudaMemcpyHostToDevice, s->stream));
  if (s->use_graph) {
    auto key = std::make_tuple(B, c.t0, c.g);
    auto it = s->graphs.find(key);
    if (it == s->graphs.end()) {
      GraphEntry ge;
      RC(build_graph(s, c, &ge));
      it = s->graphs.emplace(key, ge).first;
    }
    CK(cudaGraphLaunch(it->second.exec, s->stream));
    s->last_launches = it->second.launches;
    s->graph_launches = 1;
  } else {
    RC(enqueue(s, c, s->stream));
    s->last_launches = s->nlaunch;
    s->graph_launches = 0;
  }
  s->last_caps = c;
  for (int b = 0; b < B; ++b)
    CK(cudaMemcpyAsync(s->h_wav + (size_t)b * c.wav, s->buf.wav + (size_t)b * c.wav, (size_t)n_samples_out[b] * 4,
                       cudaMemcpyDeviceToHost, s->stream));
  CK(cudaStreamSynchronize(s->stream));
  for (int b = 0; b < B; ++b)
    memcpy(wav_out + (size_t)b * max_samples, s->h_wav + (size_t)b * c.wav, (size_t)n_samples_out[b] * 4);
  return 0;
}

extern "C" int plow_s3gen_synthesize_host(void* hp, int voice_id, const int32_t* tokens, int n_tokens,
                                          float* wav_out, int max_samples, unsigned long long seed,
                                          int* n_samples_out) {
  return plow_s3gen_synthesize_batch_host(hp, 1, &voice_id, tokens, &n_tokens, wav_out, max_samples, &seed,
                                          n_samples_out);
}

// Stats of the last synthesize call: out[0] = kernel launches in the pipeline, out[1] = CUDA
// graph launches issued (1 in graph mode, 0 direct), out[2] = graphs cached, out[3..5] = token /
// mel / generated-mel row capacity of the bucket used.
extern "C" int plow_s3gen_stats(void* hp, long long* out) {
  if (!hp || !out) return -1;
  S3* s = (S3*)hp;
  out[0] = s->last_launches;
  out[1] = s->graph_launches;
  out[2] = (long long)s->graphs.size();
  out[3] = s->last_caps.t0;
  out[4] = s->last_caps.t1;
  out[5] = s->last_caps.g;
  return 0;
}

// Test hook (not part of the plowrt ABI): copies n floats of a named internal buffer of batch
// item b from the last synthesize call to host memory. Row capacities: plow_s3gen_stats.
//   "mu" [t1][80], "noise" [t1][80] (CFM z), "mel" [g][80] (generated mel), "f0" [g],
//   "src" [480 g], "post" [h2][18], "wav" [480 g]; with PLOW_S3GEN_DEBUG=1 at create also
//   "sine_noise" [9][480 g] (unit normals) and "sine_phase" [9].
extern "C" int plow_s3gen_debug_read(void* hp, const char* name, int b, float* dst, long n) {
  if (!hp || !name || !dst) return -1;
  S3* s = (S3*)hp;
  const Caps& c = s->last_caps;
  const Buf& u = s->buf;
  const float* src = nullptr;
  long cap = 0;
  const std::string nm(name);
  if (nm == "mu") src = u.mu, cap = (long)c.t1 * 80;
  else if (nm == "noise") src = u.z, cap = (long)c.t1 * 80;
  else if (nm == "mel") src = u.melg, cap = (long)c.g * 80;
  else if (nm == "f0") src = u.f0, cap = c.g;
  else if (nm == "src") src = u.src, cap = c.wav;
  else if (nm == "post") src = u.post, cap = (long)c.h2 * 18;
  else if (nm == "wav") src = u.wav, cap = c.wav;
  else if (nm == "sine_noise" && u.dbg_noise) src = u.dbg_noise, cap = 9L * c.wav;
  else if (nm == "sine_phase" && u.dbg_phase) src = u.dbg_phase, cap = 9;
  if (!src || n > cap || b < 0 || b >= c.B) return -1;
  CK(cudaSetDevice(s->device));
  CK(cudaMemcpy(dst, src + (long)b * cap, n * 4, cudaMemcpyDeviceToHost));
  return 0;
}
