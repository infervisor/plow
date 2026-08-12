/* px8_flash_fp8pv_bench.cu — is an e4m3 P.V worth it for the hd512 FULL-attention flash prefill?
 *
 * WHY.  PX-7 Result 4 attributes 16.48 s of plow's 30.85 s 127k prefill (53%) to the hd512 flash
 * prefill, running at 128 TFLOP/s against vLLM's implied 230.  PX-7 Result 5 proposes ONE change:
 * drop to fp8-only staging, go BQ=64/BKV=32, and run BOTH mmas as e4m3 — claiming the smem budget
 * makes the bigger tiling free.  The shipped px4 fp8mma arm already runs QK as mma.m16n8k32.e4m3
 * but dequantizes V to fp16 for a mma.m16n8k16 P.V.
 *
 * WHAT PX-7 DID NOT CHECK, and what this bench exists to settle:
 *   (1) mma.m16n8k32 wants BOTH operands with the CONTRACTION dim contiguous per lane.  For P.V
 *       the contraction is kv.  P has kv contiguous (it is register-resident, we choose).  V does
 *       NOT: it is staged natural Vs8[kv][hd].  An fp8 P.V therefore needs V TRANSPOSED — the
 *       exact thing FlashAttention-3's fp8 path pays for.  Either sm_120a's 8-bit transposing
 *       ldmatrix (ldmatrix.m16n16.x2.trans.b8) hands back the right fragment for free, or a real
 *       smem transpose pass eats the win.  MODE `layout` dumps that instruction's fragment map.
 *   (2) whether the P.V mma is a big enough share of the per-tile time for 4x-ing it to matter.
 *       The px4-era ablation put P.V at 6% of the PRE-px4 T5 kernel.  If that still holds, the
 *       whole lever is worth ~5%, not 2x.
 *   (3) whether the kernel is KV-re-read-traffic bound at all (that is the entire BQ=64 argument).
 *       Per-tile time flat across seq_kv 8k..128k => it is not, and BQ=64 buys nothing.
 *
 * ARMS
 *   A  the SHIPPED d_flash_prefill_px4<512,32,16,true> called directly (no copy, zero fidelity
 *      risk): QK e4m3 k32, V dequant e4m3->fp16 with v_scale folded in, P.V m16n8k16 f16.
 *   B  new BQ=32/BKV=32: QK e4m3 k32, P quantized to e4m3, P.V m16n8k32 e4m3 straight off the
 *      RAW Vs8 tile via the transposing 8-bit ldmatrix.  No V dequant pass at all.
 *   Bg same as B but the B operand is gathered with 8 strided LDS.8 per mma instead of ldmatrix —
 *      correct by construction, so it is the ORACLE that validates B's ldmatrix fragment map.
 *
 * NUMERICS.  k_ref computes the same attention in f32 from the same e4m3 K/V, and every arm is
 * scored against it (max abs, max rel, RMS).  e4m3 P carries ~2 decimal digits and the online
 * rescale must not compound; the gate is arm B's error staying at arm A's order of magnitude.
 *
 * Run it under perf-data/tools/gpulease.  Build (needs the fp8 arms compiled in):
 *   nvcc -std=c++17 -gencode arch=compute_120a,code=sm_120a -O3 -I runtime/common -I runtime/nvidia \
 *        -DPLOW_FP8_KV=1 -Xptxas -v perf-data/px8_flash_fp8pv_bench.cu -o /tmp/px8bench
 * (-gencode, not -arch: -arch also emits a compute_120 PTX image and the 8-bit ldmatrix is
 *  sm_120a-only, so the PTX pass fails.  That is a build trap, not a capability limit.)
 */
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include "op_attention.cuh"

#if !defined(PLOW_FP8_KV)
#error "build with -DPLOW_FP8_KV=1 (the px4 fp8mma arm only exists in fp8 objects)"
#endif

#define CHK(x) do{ cudaError_t e_=(x); if(e_!=cudaSuccess){ \
  printf("CUDA ERR %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e_)); exit(1);} }while(0)

static const int HD = 512;
static const int PADB = FA_PRE_PAD;      /* 8  bf16 elems */
static const int PAD8 = FA_FP8_PAD8_K;   /* 32 bytes      */

/* ============================ 8-bit transposing ldmatrix ================================== */
/* sm_120a only.  Loads TWO 16x16 byte matrices and returns them transposed, 4 regs/lane.  The
 * fragment map is dumped by `layout` mode and asserted by arm Bg. */
#if !PLOW_NV_FA_FP8PV /* op_attention.cuh only defines it when the PX-8 arm is compiled in */
__device__ __forceinline__ void fa_ldmatrix_x2_trans_b8(unsigned (&r)[4], const void* smem) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
#endif
/* MEASURED fragment map (`layout` mode, sm_120a): lane L supplies the address of source row
 * L (rows 0..15 = matrix 0, 16..31 = matrix 1) and receives, for source column n:
 *   r0 = T[n = L>>2    ][srcrow 4*(L&3) .. +3]   (matrix 0)
 *   r1 = T[n = (L>>2)+8][srcrow 4*(L&3) .. +3]   (matrix 0)
 *   r2 = T[n = L>>2    ][srcrow 4*(L&3) .. +3]   (matrix 1)
 *   r3 = T[n = (L>>2)+8][srcrow 4*(L&3) .. +3]   (matrix 1)
 * mma.m16n8k32's B operand instead wants lane L to hold B[n=L>>2][k = 8*(L&3) .. +7] as two
 * regs.  Those differ by a QUAD PERMUTATION of k — which would cost 4 SHFLs per ldmatrix.
 * It does not have to: nothing forces smem row order to equal kv order.  Staging V's kv rows
 * in the permuted order below makes {r0,r2} and {r1,r3} EXACTLY the two B operands, for free
 * (cp.async copies whole 16B lines; only the destination row index changes).
 *   smem row s holds kv = 8*((s>>2)&3) + (s&3) + (s>=16 ? 4 : 0)
 *   i.e. kv -> s = ((kv & 4) ? 16 : 0) + 4*(kv>>3) + (kv & 3) */
__host__ __device__ __forceinline__ int px8_vrow(int kv) {
    return ((kv & 4) ? 16 : 0) + 4 * (kv >> 3) + (kv & 3);
}

__global__ void k_layout(unsigned char* out) {
    __shared__ unsigned char s[2 * 16 * 16];
    /* mat m, row r, col c  ->  byte (m*2 + (r>=8)) is NOT encodable in 8 bits together with
     * (r,c), so encode value = r*16 + c and run the two matrices with the same content; the
     * matrix a byte came from is recovered from which register pair it landed in. */
    for (int i = threadIdx.x; i < 512; i += 32) s[i] = (unsigned char)(i & 255);
    __syncwarp();
    unsigned r[4];
    /* lane L supplies the address of row L%16 of matrix L/16 */
    fa_ldmatrix_x2_trans_b8(r, &s[(threadIdx.x % 16) * 16 + (threadIdx.x / 16) * 256]);
    for (int i = 0; i < 4; i++) ((unsigned*)out)[threadIdx.x * 4 + i] = r[i];
}

/* ============================== f32 reference attention ==================================== */
/* One block per (q row, head).  K/V are e4m3 with per-kv-row f32 scales, head-major [kv][hd]. */
__global__ void k_ref(float* __restrict__ O, const __nv_bfloat16* __restrict__ Q,
                      const unsigned char* __restrict__ K, const unsigned char* __restrict__ V,
                      const float* __restrict__ ksc, const float* __restrict__ vsc,
                      unsigned seq_q, unsigned seq_kv, unsigned n_head, unsigned n_kv_head,
                      unsigned q_pos0, float scale) {
    const unsigned q = blockIdx.x, h = blockIdx.y;
    const unsigned hkv = h / (n_head / n_kv_head);
    const __nv_bfloat16* Qr = Q + ((size_t)q * n_head + h) * HD;
    const unsigned char* Kb = K + (size_t)hkv * seq_kv * HD;
    const unsigned char* Vb = V + (size_t)hkv * seq_kv * HD;
    const long qabs = (long)q_pos0 + q;
    __shared__ float part[8];
    __shared__ float acc[HD];
    __shared__ float m_s, l_s;
    for (int d = threadIdx.x; d < HD; d += blockDim.x) acc[d] = 0.f;
    if (threadIdx.x == 0) { m_s = -1e30f; l_s = 0.f; }
    __syncthreads();
    for (unsigned kv = 0; kv <= (unsigned)qabs && kv < seq_kv; kv++) {
        float d = 0.f;
        for (int i = threadIdx.x; i < HD; i += blockDim.x)
            d += __bfloat162float(Qr[i]) * fp8_to_f32(Kb[(size_t)kv * HD + i]);
        d = warp_sum32(d);
        if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = d;
        __syncthreads();
        float s = 0.f;
        for (unsigned w = 0; w < blockDim.x / 32; w++) s += part[w];
        s = s * ksc[kv] * scale;
        const float mn = fmaxf(m_s, s);
        const float corr = (m_s < -1e29f) ? 0.f : expf(m_s - mn);
        const float p = expf(s - mn) * vsc[kv];
        const float pl = expf(s - mn);
        __syncthreads();
        for (int i = threadIdx.x; i < HD; i += blockDim.x)
            acc[i] = acc[i] * corr + p * fp8_to_f32(Vb[(size_t)kv * HD + i]);
        if (threadIdx.x == 0) { l_s = l_s * corr + pl; m_s = mn; }
        __syncthreads();
    }
    const float inv = (l_s > 0.f) ? 1.f / l_s : 0.f;
    for (int i = threadIdx.x; i < HD; i += blockDim.x)
        O[((size_t)q * n_head + h) * HD + i] = acc[i] * inv;
}

/* ================================ ARM A — the shipped kernel =============================== */
__global__ void __launch_bounds__(256, 1)
k_armA(__nv_bfloat16* O, const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
       const float* ksc, const float* vsc, unsigned seq_q, unsigned seq_kv, unsigned n_head,
       unsigned n_kv_head, unsigned q_pos0, unsigned kv_stride, unsigned kv_mask, float scale) {
    extern __shared__ float sm[];
    d_flash_prefill_px4<512, 32, 16, true>(nullptr, nullptr, Q, K, V, O, seq_q, seq_kv, n_head,
                                           n_kv_head, q_pos0, /*window*/ 0, /*nsplit*/ 1,
                                           kv_stride, kv_mask, scale, blockIdx.x, gridDim.x, sm,
                                           ksc, vsc);
}

/* ================================ ARM B — e4m3 P.V, BKV=32 ================================= */
/* Same skeleton as d_flash_prefill_px4 (cp.async ring, register softmax fused into the P.V A
 * fragment, SsA/SsB hd-half split) with three changes:
 *   - BKV 16 -> 32, so the P.V contraction fills a k32;
 *   - the V dequant pass is GONE.  V stays raw e4m3 in Vs8 and the B operand comes out of the
 *     transposing 8-bit ldmatrix;
 *   - P is quantized to e4m3.  v_scale is per-kv-row so it must fold into P (it cannot fold into
 *     a raw e4m3 V), and P*v_scale would sit far below e4m3's normal floor — so P is normalised
 *     by the tile's max v_scale and the accumulator is un-normalised at the epilogue.  That
 *     per-tile vmax is a 8xLDS.128-broadcast + 31 FMAX, cheaper than the dequant it replaces.
 *
 * ABL bits (bench-only): 1 skip QK mma, 2 skip softmax, 4 skip P.V, 8 skip cp.async issue.
 * PVMODE: 0 = ldmatrix.m16n16.x2.trans.b8, 1 = 8x LDS.8 gather (the correctness oracle). */
#define PX8_BQ 32
#define PX8_BKV 32
/* smem floats: mbar(4) + SsA/SsB + Qs bf16 + qsc + ksc + vsc + Qs8 + Ks8 + Vs8 */
#define PX8_SMEM_FLOATS                                                                             \
    (4 + 2 * PX8_BQ * PX8_BKV + (PX8_BQ * (HD + PADB) + 1) / 2 + PX8_BQ + 2 * PX8_BKV +              \
     (PX8_BQ * (HD + PAD8) + 3) / 4 + 2 * ((PX8_BKV * (HD + PAD8) + 3) / 4))

template <int ABL, int PVMODE>
__device__ void d_px8_armB(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
                           const __nv_bfloat16* __restrict__ V, __nv_bfloat16* __restrict__ O,
                           unsigned seq_q, unsigned seq_kv, unsigned n_head, unsigned n_kv_head,
                           unsigned q_pos0, unsigned kv_stride, unsigned kv_mask, float scale,
                           unsigned slice, unsigned nblk, float* lds,
                           const float* __restrict__ k_scale, const float* __restrict__ v_scale) {
    constexpr int BQ = PX8_BQ, BKV = PX8_BKV;
    constexpr int WPV_N = 4, HDW = HD / WPV_N, NJ_PV = HDW / 8; /* 128, 16 */
    constexpr int KSTEPS8_H = HD / 2 / 32;                      /* 8 k32 steps per hd half */
    constexpr int HCH8 = HD / 16;                               /* 16B cp.async lines per e4m3 row */
    /* P is quantised as  p * (vsc/vmax) * PS ; the accumulator carries vmax/PS. */
    constexpr float PS = 256.0f;

    float* SsA = lds + 4;
    float* SsB = SsA + BQ * BKV;
    __nv_bfloat16* Qs = (__nv_bfloat16*)(SsB + BQ * BKV);
    float* qsc_s = (float*)(Qs + BQ * (HD + PADB));
    float* ksc_s = qsc_s + BQ;
    float* vsc_s = ksc_s + BKV;
    unsigned char* Qs8 = (unsigned char*)(vsc_s + BKV);
    unsigned char* Ks8 = Qs8 + BQ * (HD + PAD8);
    unsigned char* Vs8 = Ks8 + BKV * (HD + PAD8);

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const unsigned gqa = n_head / n_kv_head;
    const unsigned n_qt = (seq_q + BQ - 1) / BQ;
    const unsigned n_work = n_qt * n_head;
    const float lscale = FA_SCALE(scale);
    /* QK: 8 warps = 2(hd half) x 2(query 16-rows) x 2(kv 16-col group, 2 n8 subtiles each) */
    const int qk_kh = warp >> 2, qk_wm = (warp >> 1) & 1, qk_wn = warp & 1;
    const int pv_wm = warp >> 2, pv_wn = warp & 3;
    const int r0 = pv_wm * 16 + (lane >> 2);   /* this lane's softmax rows: r0 and r0+8 */
    const int kb0 = (lane & 3) * 8;            /* this lane's 8 CONSECUTIVE kv columns */

    for (unsigned w = slice; w < n_work; w += nblk) {
        const unsigned h = w % n_head;
        const unsigned qt = w / n_head;
        const unsigned q0 = qt * BQ;
        const unsigned hkv = h / gqa;
        const unsigned hi = seq_kv;

        const __nv_bfloat16* Qh = Q + (size_t)q0 * n_head * HD + (size_t)h * HD;
        const unsigned char* Kb8 = (const unsigned char*)K + (size_t)hkv * kv_stride * HD;
        const unsigned char* Vb8 = (const unsigned char*)V + (size_t)hkv * kv_stride * HD;
        const float* ksc = k_scale + (size_t)hkv * kv_stride;
        const float* vsc = v_scale + (size_t)hkv * kv_stride;

        __syncthreads();
        for (int idx = tid; idx < BQ * HD; idx += 256) {
            int r = idx / HD, c = idx % HD;
            __nv_bfloat16 v = __float2bfloat16(0.f);
            if (q0 + r < seq_q) v = Qh[(size_t)r * n_head * HD + c];
            Qs[r * (HD + PADB) + c] = v;
        }
        __syncthreads();
        /* Q -> e4m3 once per q-tile, stored in mma A-fragment order (px4's layout, verbatim). */
        {
            const int qr = warp * (BQ / 8) + (lane >> 3);
            const int le = lane & 7;
            const int cb = le * (HD / 8);
            float amax = 0.0f;
            for (int e = 0; e < HD / 8; e++)
                amax = fmaxf(amax, fabsf(__bfloat162float(Qs[qr * (HD + PADB) + cb + e])));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
            const float qinv = (amax > 0.0f) ? (PLOW_FP8_E4M3_MAX / amax) : 0.0f;
            if (le == 0) qsc_s[qr] = amax * (1.0f / PLOW_FP8_E4M3_MAX);
            const int qm = qr >> 4, r7 = qr & 7, hi8 = ((qr >> 3) & 1) * 8;
            for (int g = 0; g < HD / 64; g++) {
                const int c = cb + g * 8;
                unsigned u0 = 0, u1 = 0;
#pragma unroll
                for (int j = 0; j < 4; j++)
                    u0 |= (unsigned)quant_fp8(__bfloat162float(Qs[qr * (HD + PADB) + c + j]) * qinv)
                          << (8 * j);
#pragma unroll
                for (int j = 0; j < 4; j++)
                    u1 |= (unsigned)quant_fp8(
                              __bfloat162float(Qs[qr * (HD + PADB) + c + 4 + j]) * qinv)
                          << (8 * j);
                const int kh = c >> 8, kf = (c & 255) >> 5, L = r7 * 4 + ((c >> 3) & 3);
                *(uint2*)&Qs8[(((qm * 2 + kh) * 8 + kf) << 9) + L * 16 + hi8] = make_uint2(u0, u1);
            }
        }

        float oacc[NJ_PV][4];
#pragma unroll
        for (int nj = 0; nj < NJ_PV; nj++)
#pragma unroll
            for (int e = 0; e < 4; e++) oacc[nj][e] = 0.0f;
        float m_reg[2] = {FA_NEG_INF, FA_NEG_INF}, l_reg[2] = {0.0f, 0.0f};
        float gscale = 0.0f; /* the accumulator's current units (see the P normaliser below) */

        const int qabs_max = (int)(q_pos0 + q0 + BQ - 1);
        long cap = (long)hi - 1;
        if ((long)qabs_max < cap) cap = (long)qabs_max;
        const int nt = (cap >= 0) ? (int)(cap / BKV) + 1 : 0;

        auto stageK = [&](unsigned kv0) {
            if constexpr (!(ABL & 8)) {
                for (int L = tid; L < BKV * HCH8; L += 256) {
                    int r = L / HCH8, c16 = (L % HCH8) * 16;
                    unsigned kv = kv0 + (unsigned)r;
                    bool in = (kv < hi);
                    const unsigned char* g = in ? Kb8 + (size_t)(kv & kv_mask) * HD + c16 : Kb8;
                    fa_cp_async_cg16(&Ks8[r * (HD + PAD8) + c16], g, in ? 16 : 0);
                }
            } else { (void)kv0; }
            fa_cp_commit();
        };
        auto stageV = [&](unsigned kv0) {
            /* kv row r lands in smem row px8_vrow(r): the free half of the fp8 P.V. */
            if constexpr (!(ABL & 8)) {
                for (int L = tid; L < BKV * HCH8; L += 256) {
                    int r = L / HCH8, c16 = (L % HCH8) * 16;
                    unsigned kv = kv0 + (unsigned)r;
                    bool in = (kv < hi);
                    const unsigned char* g = in ? Vb8 + (size_t)(kv & kv_mask) * HD + c16 : Vb8;
                    fa_cp_async_cg16(&Vs8[px8_vrow(r) * (HD + PAD8) + c16], g, in ? 16 : 0);
                }
            } else { (void)kv0; }
            fa_cp_commit();
        };

        __syncthreads();
        float sc_pf = 0.0f;
        if (tid < 2 * BKV && nt > 0) {
            const unsigned kvr = (unsigned)(tid & (BKV - 1));
            if (kvr < hi) sc_pf = (tid < BKV) ? ksc[kvr & kv_mask] : vsc[kvr & kv_mask];
        }
        if (nt > 0) stageK(0);

        for (int t = 0; t < nt; t++) {
            const unsigned kv0 = (unsigned)t * BKV;
            stageV(kv0);
            if (tid < 2 * BKV) {
                ((tid < BKV) ? ksc_s : vsc_s)[tid & (BKV - 1)] = sc_pf;
                const unsigned kvn = kv0 + BKV + (unsigned)(tid & (BKV - 1));
                sc_pf = 0.0f;
                if (kvn < hi) sc_pf = (tid < BKV) ? ksc[kvn & kv_mask] : vsc[kvn & kv_mask];
            }
            fa_cp_wait<1>();
            __syncthreads();

            /* ---- S = Q.K^T, e4m3 k32.  Each warp: one hd half x 16 query x 16 kv (2 n8). ---- */
            {
                float acc[2][4];
#pragma unroll
                for (int j = 0; j < 2; j++)
#pragma unroll
                    for (int e = 0; e < 4; e++) acc[j][e] = 0.f;
                if constexpr (!(ABL & 1)) {
                const int khoff = qk_kh * (HD / 2);
                const int kbq = khoff + 8 * (lane & 3);
                const uint4* QsAf = (const uint4*)&Qs8[(((qk_wm * 2 + qk_kh) * 8) << 9) + lane * 16];
                float accB[2][4];
#pragma unroll
                for (int j = 0; j < 2; j++)
#pragma unroll
                    for (int e = 0; e < 4; e++) accB[j][e] = 0.f;
#pragma unroll
                for (int kf = 0; kf < KSTEPS8_H; kf++) {
                    const int kb = kbq + kf * 32;
                    unsigned a8[4];
                    const uint4 av = QsAf[kf * 32];
                    a8[0] = av.x; a8[2] = av.y; a8[1] = av.z; a8[3] = av.w;
#pragma unroll
                    for (int j = 0; j < 2; j++) {
                        const int nn = qk_wn * 16 + j * 8 + (lane >> 2);
                        const uint2 bb = *(const uint2*)&Ks8[nn * (HD + PAD8) + kb];
                        unsigned b8[2] = {bb.x, bb.y};
                        if (kf & 1) fa_mma_fp8_k32(accB[j], a8, b8, accB[j]);
                        else        fa_mma_fp8_k32(acc[j], a8, b8, acc[j]);
                    }
                }
#pragma unroll
                for (int j = 0; j < 2; j++)
#pragma unroll
                    for (int e = 0; e < 4; e++) acc[j][e] += accB[j][e];
                }
                float* Sdst = qk_kh ? SsB : SsA;
                const int qlo = qk_wm * 16 + (lane / 4);
                const float q0s = qsc_s[qlo], q1s = qsc_s[qlo + 8];
#pragma unroll
                for (int j = 0; j < 2; j++) {
                    const int kc0 = qk_wn * 16 + j * 8 + (lane % 4) * 2;
                    const float ks0 = ksc_s[kc0] * lscale, ks1 = ksc_s[kc0 + 1] * lscale;
                    *(float2*)&Sdst[qlo * BKV + kc0] =
                        make_float2(acc[j][0] * ks0 * q0s, acc[j][1] * ks1 * q0s);
                    *(float2*)&Sdst[(qlo + 8) * BKV + kc0] =
                        make_float2(acc[j][2] * ks0 * q1s, acc[j][3] * ks1 * q1s);
                }
            }
            __syncthreads(); /* Ss published; Ks8 free for K[t+1] */
            if (t + 1 < nt) stageK(kv0 + BKV); else fa_cp_commit();

            const unsigned rmax = (hi - kv0 < (unsigned)BKV) ? (hi - kv0) : (unsigned)BKV;
            fa_cp_wait<1>();  /* V[t] */

            /* ---- register softmax fused into the e4m3 P.V A fragment ---- */
            unsigned af_pv[4];
            float vmax = 0.0f;
            {
                /* tile-wide max v_scale: broadcast reads, no barrier (vsc_s written pre-barrier) */
#pragma unroll
                for (int i = 0; i < BKV; i += 4) {
                    const float4 v4 = *(const float4*)&vsc_s[i];
                    vmax = fmaxf(fmaxf(vmax, v4.x), fmaxf(fmaxf(v4.y, v4.z), v4.w));
                }
                const float vnorm = (vmax > 0.0f) ? (PS / vmax) : 0.0f;
                float p[2][8];
                float corr[2];
#pragma unroll
                for (int j = 0; j < 2; j++) {
                    const int row = r0 + j * 8;
                    const int qabs = (int)(q_pos0 + q0 + row);
                    float s[8], mx = FA_NEG_INF;
                    if constexpr (!(ABL & 2)) {
                    const float4 a0 = *(const float4*)&SsA[row * BKV + kb0];
                    const float4 a1 = *(const float4*)&SsA[row * BKV + kb0 + 4];
                    const float4 b0 = *(const float4*)&SsB[row * BKV + kb0];
                    const float4 b1 = *(const float4*)&SsB[row * BKV + kb0 + 4];
                    s[0] = a0.x + b0.x; s[1] = a0.y + b0.y; s[2] = a0.z + b0.z; s[3] = a0.w + b0.w;
                    s[4] = a1.x + b1.x; s[5] = a1.y + b1.y; s[6] = a1.z + b1.z; s[7] = a1.w + b1.w;
#pragma unroll
                    for (int ci = 0; ci < 8; ci++) {
                        const int col = kb0 + ci;
                        const int kv = (int)kv0 + col;
                        if ((unsigned)col >= rmax || kv > qabs) s[ci] = FA_NEG_INF;
                        mx = fmaxf(mx, s[ci]);
                    }
                    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
                    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
                    const float m_new = fmaxf(m_reg[j], mx);
                    corr[j] = (m_reg[j] == FA_NEG_INF) ? 0.0f : FA_EXP(m_reg[j] - m_new);
                    float lsum = 0.0f;
#pragma unroll
                    for (int ci = 0; ci < 8; ci++) {
                        p[j][ci] = (s[ci] == FA_NEG_INF || m_new == FA_NEG_INF)
                                       ? 0.0f : FA_EXP(s[ci] - m_new);
                        lsum += p[j][ci];
                    }
                    lsum += __shfl_xor_sync(0xffffffffu, lsum, 1);
                    lsum += __shfl_xor_sync(0xffffffffu, lsum, 2);
                    l_reg[j] = l_reg[j] * corr[j] + lsum;
                    m_reg[j] = m_new;
                    /* v_scale folds into P (it cannot fold into a RAW e4m3 V), normalised by the
                     * tile vmax so the quantised values sit in e4m3's normal range. */
#pragma unroll
                    for (int ci = 0; ci < 8; ci++) p[j][ci] *= vsc_s[kb0 + ci] * vnorm;
                    } else {
                    corr[j] = 1.0f;
#pragma unroll
                    for (int ci = 0; ci < 8; ci++) { s[ci] = 0.f; p[j][ci] = 0.f; }
                    (void)qabs; (void)mx;
                    }
                }
                /* A fragment: a0=(r0,k0..3) a1=(r0+8,k0..3) a2=(r0,k4..7) a3=(r0+8,k4..7) */
                unsigned u[4] = {0, 0, 0, 0};
#pragma unroll
                for (int b = 0; b < 4; b++) {
                    u[0] |= (unsigned)quant_fp8(p[0][b]) << (8 * b);
                    u[1] |= (unsigned)quant_fp8(p[1][b]) << (8 * b);
                    u[2] |= (unsigned)quant_fp8(p[0][4 + b]) << (8 * b);
                    u[3] |= (unsigned)quant_fp8(p[1][4 + b]) << (8 * b);
                }
                af_pv[0] = u[0]; af_pv[1] = u[1]; af_pv[2] = u[2]; af_pv[3] = u[3];
                /* The tile's P normaliser (vmax/PS) must NOT multiply the 64 mma outputs — that
                 * would hand arm B 128 FP ops/lane/tile arm A never pays.  Instead carry it in
                 * the accumulator's UNITS: oacc is O/gscale, and the units change folds into the
                 * online-softmax corr that is already applied.  Epilogue multiplies by gscale. */
                const float sc_pv = (vmax > 0.0f) ? (vmax / PS) : 0.0f;
                float cadj = 1.0f;
                if (sc_pv > 0.0f) { cadj = (gscale > 0.0f) ? (gscale / sc_pv) : 0.0f; gscale = sc_pv; }
                const float c0 = corr[0] * cadj, c1 = corr[1] * cadj;
#pragma unroll
                for (int nj = 0; nj < NJ_PV; nj++) {
                    oacc[nj][0] *= c0; oacc[nj][1] *= c0;
                    oacc[nj][2] *= c1; oacc[nj][3] *= c1;
                }
            }
            __syncthreads(); /* every thread's V[t] bytes visible to every P.V warp */

            /* ---- O += P.V, e4m3 k32, straight off the RAW Vs8 tile ---- */
            if constexpr (!(ABL & 4)) {
            if (PVMODE == 0) {
#pragma unroll
                for (int nj = 0; nj < NJ_PV; nj += 2) {
                    unsigned rb[4];
                    /* lane L addresses kv row L of the 32x16 byte block at hd = pv_wn*HDW+nj*8 */
                    fa_ldmatrix_x2_trans_b8(rb, &Vs8[lane * (HD + PAD8) + pv_wn * HDW + nj * 8]);
                    unsigned b0[2] = {rb[0], rb[2]}, b1[2] = {rb[1], rb[3]};
                    fa_mma_fp8_k32(oacc[nj], af_pv, b0, oacc[nj]);
                    fa_mma_fp8_k32(oacc[nj + 1], af_pv, b1, oacc[nj + 1]);
                }
            } else {
#pragma unroll
                for (int nj = 0; nj < NJ_PV; nj++) {
                    const int hdc = pv_wn * HDW + nj * 8 + (lane >> 2);
                    unsigned b8[2] = {0, 0};
#pragma unroll
                    for (int b = 0; b < 4; b++) {
                        b8[0] |= (unsigned)Vs8[px8_vrow(kb0 + b) * (HD + PAD8) + hdc] << (8 * b);
                        b8[1] |= (unsigned)Vs8[px8_vrow(kb0 + 4 + b) * (HD + PAD8) + hdc]
                                 << (8 * b);
                    }
                    fa_mma_fp8_k32(oacc[nj], af_pv, b8, oacc[nj]);
                }
            }
            }
            __syncthreads(); /* P.V done reading Vs8 before V[t+1] restages it */
        }

#pragma unroll
        for (int nj = 0; nj < NJ_PV; nj++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const unsigned qrow = (unsigned)(r0 + (e >> 1) * 8);
                const unsigned qabs_row = q0 + qrow;
                if (qabs_row >= seq_q) continue;
                const int hd = pv_wn * HDW + nj * 8 + (lane & 3) * 2 + (e & 1);
                const float lv = l_reg[e >> 1];
                const float inv = (lv > 0.0f) ? (gscale / lv) : 0.0f;
                O[(size_t)(qabs_row * n_head + h) * HD + hd] = __float2bfloat16(oacc[nj][e] * inv);
            }
    }
}

/* Bp — the SHIPPED d_flash_prefill_px8 called directly (needs -DPLOW_NV_FA_FP8PV=1).  Arm B above
 * is a bench-local copy carrying the ablation bits and the LDS.8 oracle; Bp is what serving would
 * actually run, so it is the one whose numerics and timing are load-bearing. */
#if PLOW_NV_FA_FP8PV
__global__ void __launch_bounds__(256, 1)
k_armBp(__nv_bfloat16* O, const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
        const float* ksc, const float* vsc, unsigned seq_q, unsigned seq_kv, unsigned n_head,
        unsigned n_kv_head, unsigned q_pos0, unsigned kv_stride, unsigned kv_mask, float scale) {
    extern __shared__ float sm[];
    d_flash_prefill_px8<512, 32, 32, true>(nullptr, nullptr, Q, K, V, O, seq_q, seq_kv, n_head,
                                           n_kv_head, q_pos0, /*window*/ 0, /*nsplit*/ 1,
                                           kv_stride, kv_mask, scale, blockIdx.x, gridDim.x, sm,
                                           ksc, vsc);
}
#endif

template <int ABL, int PVMODE>
__global__ void __launch_bounds__(256, 1)
k_armB(__nv_bfloat16* O, const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
       const float* ksc, const float* vsc, unsigned seq_q, unsigned seq_kv, unsigned n_head,
       unsigned n_kv_head, unsigned q_pos0, unsigned kv_stride, unsigned kv_mask, float scale) {
    extern __shared__ float sm[];
    d_px8_armB<ABL, PVMODE>(Q, K, V, O, seq_q, seq_kv, n_head, n_kv_head, q_pos0, kv_stride,
                            kv_mask, scale, blockIdx.x, gridDim.x, sm, ksc, vsc);
}

/* ============================== P.V PHASE microbench ======================================= */
/* Per-32-kv P.V cost in isolation: everything the two arms actually pay between "V[t] has landed
 * as raw e4m3 in smem" and "O accumulated".  MODE 0 reproduces arm A (dequant e4m3->fp16 with the
 * v_scale folded, then 2x16 ldmatrix.x2.trans.b16 + m16n8k16 f16 mma); MODE 1 is arm B's
 * ldmatrix.m16n16.x2.trans.b8 + m16n8k32 e4m3; MODE 2 is arm B with the LDS.8 gather. */
/* VPAD is the Vs8 row pad in BYTES.  It matters: the 8-bit ldmatrix reads 16B per lane-supplied
 * ROW address, so the bank spread is (HD+VPAD)/4 mod 32 — 8 with the shipped pad of 32 (4-way
 * conflict over 16 rows), 4 with a pad of 16 (2-way).  The shipped pad was tuned for the QK
 * uint2 reads of Ks8, which is a different access; V can carry its own pad.
 * NJ is the P.V accumulator depth = BQ*HD/(256*8): 16 at BQ=32, 32 at BQ=64. */
template <int MODE, int VPAD = 32, int NJ = 16>
__global__ void __launch_bounds__(256, 1) k_pvphase(float* sink, int iters) {
    /* dynamic smem: MODE 0 needs Vs8 + a bf16 Vs mirror = 50,816 B, over the 48 KiB STATIC cap */
    extern __shared__ unsigned char smem_pv[];
    unsigned char* Vs8 = smem_pv;
    __nv_bfloat16* Vs = (__nv_bfloat16*)(smem_pv + 32 * (HD + VPAD));
    float* vsc_s = (float*)(Vs + 32 * (HD + PADB));
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    constexpr int NJ_PV = NJ, WPV_N = (HD / 8) / NJ, HDW = HD / WPV_N;
    const int pv_wn = warp & (WPV_N - 1);
    for (int i = tid; i < 32 * (HD + VPAD); i += 256) Vs8[i] = (unsigned char)(i * 7 + 3);
    for (int i = tid; i < 32; i += 256) vsc_s[i] = 0.01f + 0.001f * i;
    __syncthreads();
    unsigned af[4] = {0x3c003c00u ^ (unsigned)lane, 0x38003800u, 0x34003400u, 0x30003000u};
    float oacc[NJ_PV][4];
#pragma unroll
    for (int nj = 0; nj < NJ_PV; nj++)
#pragma unroll
        for (int e = 0; e < 4; e++) oacc[nj][e] = 0.f;
    const int kb0 = (lane & 3) * 8;
    for (int it = 0; it < iters; it++) {
        if (tid == 0) Vs8[it & 1023] = (unsigned char)it; /* defeat hoisting */
        __syncthreads();
        if (MODE == 0) {
            /* arm A: two 16-kv sub-tiles, each dequanted then consumed by 16 f16 mma */
#pragma unroll
            for (int half = 0; half < 2; half++) {
                constexpr int HCH8 = HD / 16;
                for (int L = tid; L < 16 * HCH8; L += 256) {
                    const int r = half * 16 + L / HCH8, c16 = (L % HCH8) * 16;
                    const __half2 vs2 = __float2half2_rn(vsc_s[r]);
                    const uint4 raw = *(const uint4*)&Vs8[r * (HD + VPAD) + c16];
                    uint4 o;
                    __half2 h2;
#define CVT8(dst, wv)                                                                               \
    {                                                                                               \
        __half2_raw h0 = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)((wv) & 0xffffu),        \
                                                    __NV_E4M3);                                     \
        __half2_raw h1 = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)((wv) >> 16), __NV_E4M3);\
        h2 = __hmul2(*(__half2*)&h0, vs2); (dst).x = *(unsigned*)&h2;                                \
        h2 = __hmul2(*(__half2*)&h1, vs2); (dst).y = *(unsigned*)&h2;                                \
    }
                    uint2 lo, hi2;
                    CVT8(lo, raw.x); CVT8(hi2, raw.y);
                    o.x = lo.x; o.y = lo.y; o.z = hi2.x; o.w = hi2.y;
                    *(uint4*)&Vs[r * (HD + PADB) + c16] = o;
                    CVT8(lo, raw.z); CVT8(hi2, raw.w);
                    o.x = lo.x; o.y = lo.y; o.z = hi2.x; o.w = hi2.y;
                    *(uint4*)&Vs[r * (HD + PADB) + c16 + 8] = o;
#undef CVT8
                }
                __syncthreads();
#pragma unroll
                for (int nj = 0; nj < NJ_PV; nj++) {
                    unsigned bf[2];
                    fa_ldmatrix_x2_trans(bf, &Vs[(half * 16 + (lane % 16)) * (HD + PADB) +
                                                 pv_wn * HDW + nj * 8]);
                    fa_mma_f16(oacc[nj], af, bf, oacc[nj]);
                }
            }
        } else if (MODE == 1) {
#pragma unroll
            for (int nj = 0; nj < NJ_PV; nj += 2) {
                unsigned rb[4];
                fa_ldmatrix_x2_trans_b8(rb, &Vs8[lane * (HD + VPAD) + pv_wn * HDW + nj * 8]);
                unsigned b0[2] = {rb[0], rb[2]}, b1[2] = {rb[1], rb[3]};
                fa_mma_fp8_k32(oacc[nj], af, b0, oacc[nj]);
                fa_mma_fp8_k32(oacc[nj + 1], af, b1, oacc[nj + 1]);
            }
        } else {
#pragma unroll
            for (int nj = 0; nj < NJ_PV; nj++) {
                const int hdc = pv_wn * HDW + nj * 8 + (lane >> 2);
                unsigned b8[2] = {0, 0};
#pragma unroll
                for (int b = 0; b < 4; b++) {
                    b8[0] |= (unsigned)Vs8[(kb0 + b) * (HD + VPAD) + hdc] << (8 * b);
                    b8[1] |= (unsigned)Vs8[(kb0 + 4 + b) * (HD + VPAD) + hdc] << (8 * b);
                }
                fa_mma_fp8_k32(oacc[nj], af, b8, oacc[nj]);
            }
        }
        __syncthreads();
    }
    float s = 0.f;
#pragma unroll
    for (int nj = 0; nj < NJ_PV; nj++)
#pragma unroll
        for (int e = 0; e < 4; e++) s += oacc[nj][e];
    if (s == 12345.678f) *sink = s;
}

/* ==================================== host harness ========================================= */
static uint32_t rng = 987654321u;
static uint32_t xr() { rng = rng * 1664525u + 1013904223u; return rng; }

struct Data {
    __nv_bfloat16 *Q, *O;
    unsigned char *K, *V;
    float *ksc, *vsc, *ref;
    unsigned seq_q, seq_kv, nh, nkv, q_pos0;
};

static Data make(unsigned seq_q, unsigned seq_kv, unsigned nh, unsigned nkv, unsigned q_pos0) {
    Data d{};
    d.seq_q = seq_q; d.seq_kv = seq_kv; d.nh = nh; d.nkv = nkv; d.q_pos0 = q_pos0;
    const size_t nQ = (size_t)seq_q * nh * HD, nKV = (size_t)nkv * seq_kv * HD;
    std::vector<uint16_t> hq(nQ);
    for (size_t i = 0; i < nQ; i++) {           /* bf16 in ~[-1,1] */
        float f = ((float)(xr() >> 8) / 8388608.0f) * 2.0f - 1.0f;
        hq[i] = (uint16_t)(*(unsigned*)&f >> 16);
    }
    std::vector<unsigned char> hk(nKV), hv(nKV);
    for (size_t i = 0; i < nKV; i++) {
        hk[i] = (unsigned char)(xr() & 0x6fu);  /* never 0x7f/0xff (NaN) */
        hv[i] = (unsigned char)(xr() & 0x6fu);
    }
    std::vector<float> hs((size_t)nkv * seq_kv);
    for (size_t i = 0; i < hs.size(); i++) hs[i] = 0.008f + 0.004f * ((float)(xr() & 255) / 255.f);
    CHK(cudaMalloc(&d.Q, nQ * 2)); CHK(cudaMemcpy(d.Q, hq.data(), nQ * 2, cudaMemcpyHostToDevice));
    CHK(cudaMalloc(&d.O, nQ * 2)); CHK(cudaMemset(d.O, 0, nQ * 2));
    CHK(cudaMalloc(&d.K, nKV)); CHK(cudaMemcpy(d.K, hk.data(), nKV, cudaMemcpyHostToDevice));
    CHK(cudaMalloc(&d.V, nKV)); CHK(cudaMemcpy(d.V, hv.data(), nKV, cudaMemcpyHostToDevice));
    CHK(cudaMalloc(&d.ksc, hs.size() * 4));
    CHK(cudaMemcpy(d.ksc, hs.data(), hs.size() * 4, cudaMemcpyHostToDevice));
    for (size_t i = 0; i < hs.size(); i++) hs[i] = 0.006f + 0.006f * ((float)(xr() & 255) / 255.f);
    CHK(cudaMalloc(&d.vsc, hs.size() * 4));
    CHK(cudaMemcpy(d.vsc, hs.data(), hs.size() * 4, cudaMemcpyHostToDevice));
    CHK(cudaMalloc(&d.ref, nQ * 4)); CHK(cudaMemset(d.ref, 0, nQ * 4));
    return d;
}
static void freeData(Data& d) {
    cudaFree(d.Q); cudaFree(d.O); cudaFree(d.K); cudaFree(d.V);
    cudaFree(d.ksc); cudaFree(d.vsc); cudaFree(d.ref);
}

/* max abs / max rel / rms of a bf16 output against the f32 reference */
static void score(const char* label, const Data& d) {
    const size_t n = (size_t)d.seq_q * d.nh * HD;
    std::vector<uint16_t> ho(n);
    std::vector<float> hr(n);
    CHK(cudaMemcpy(ho.data(), d.O, n * 2, cudaMemcpyDeviceToHost));
    CHK(cudaMemcpy(hr.data(), d.ref, n * 4, cudaMemcpyDeviceToHost));
    double mabs = 0, mrel = 0, se = 0, sr = 0;
    double rmax = 0;
    for (size_t i = 0; i < n; i++) if (fabs(hr[i]) > rmax) rmax = fabs(hr[i]);
    for (size_t i = 0; i < n; i++) {
        unsigned u = (unsigned)ho[i] << 16;
        float got = *(float*)&u, want = hr[i];
        double e = fabs((double)got - want);
        if (e > mabs) mabs = e;
        if (fabs(want) > 0.05 * rmax && e / fabs(want) > mrel) mrel = e / fabs(want);
        se += e * e; sr += (double)want * want;
    }
    printf("  %-34s max|e|=%.3e  max rel(>5%% of peak)=%.3e  rms/rms_ref=%.3e\n", label, mabs,
           mrel, sqrt(se / n) / sqrt(sr / n));
}

template <typename F>
static double timeit(F f, int warm, int iters) {
    for (int i = 0; i < warm; i++) f();
    CHK(cudaDeviceSynchronize());
    cudaEvent_t a, b; CHK(cudaEventCreate(&a)); CHK(cudaEventCreate(&b));
    CHK(cudaEventRecord(a));
    for (int i = 0; i < iters; i++) f();
    CHK(cudaEventRecord(b)); CHK(cudaEventSynchronize(b));
    float ms = 0; CHK(cudaEventElapsedTime(&ms, a, b));
    CHK(cudaGetLastError());
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms / iters;
}

/* KV tiles a trailing chunk actually walks, at a given BKV — the per-tile normaliser. */
static double tile_count(unsigned seq_q, unsigned seq_kv, unsigned nh, int BQ, int BKV) {
    const unsigned q_pos0 = (seq_kv > seq_q) ? (seq_kv - seq_q) : 0u;
    double tiles = 0;
    for (unsigned qt = 0; qt < (seq_q + BQ - 1) / BQ; qt++) {
        long cap = (long)seq_kv - 1;
        long qm = (long)q_pos0 + (long)qt * BQ + BQ - 1;
        if (qm < cap) cap = qm;
        if (cap >= 0) tiles += (double)(cap / BKV + 1);
    }
    return tiles * nh;
}

static void run_layout() {
    unsigned char* dv; CHK(cudaMalloc(&dv, 4096)); CHK(cudaMemset(dv, 0, 4096));
    k_layout<<<1, 32>>>(dv);
    CHK(cudaDeviceSynchronize());
    std::vector<unsigned char> h(4096);
    CHK(cudaMemcpy(h.data(), dv, 4096, cudaMemcpyDeviceToHost));
    printf("== ldmatrix.m16n16.x2.trans.b8 fragment map.  source s[row*16+col]=row*16+col;\n");
    printf("   lane L gave &s[(L%%16)*16 + (L/16)*256].  Each entry is the (row,col) of the\n");
    printf("   SOURCE byte.  For an mma.m16n8k32 B operand we need lane L to hold\n");
    printf("   B[n=L>>2][k=8*(L&3)..+7] i.e. transposed: source col = L>>2 (+8), source row =\n");
    printf("   8*(L&3)..+7 within the matrix.\n");
    for (int L = 0; L < 32; L++) {
        printf("  lane %2d:", L);
        for (int r = 0; r < 4; r++) {
            printf("  r%d[", r);
            for (int b = 0; b < 4; b++) {
                int v = h[L * 16 + r * 4 + b];
                printf("%s(%2d,%2d)", b ? "," : "", v / 16, v % 16);
            }
            printf("]");
        }
        printf("\n");
    }
    cudaFree(dv);
}

int main(int argc, char** argv) {
    const char* mode = (argc > 1) ? argv[1] : "all";
    cudaDeviceProp pr; CHK(cudaGetDeviceProperties(&pr, 0));
    const int P = pr.multiProcessorCount;
    printf("# %s  SMs=%d  L2=%.0f MiB  smemOptin=%zu B\n", pr.name, P, pr.l2CacheSize / 1048576.0,
           (size_t)pr.sharedMemPerBlockOptin);
    const size_t smemA = (size_t)FA_PX4_SMEM_FLOATS(512, 32, 16) * sizeof(float);
    const size_t smemB = (size_t)PX8_SMEM_FLOATS * sizeof(float);
#if PLOW_NV_FA_FP8PV
    const size_t smemBp = (size_t)FA_PX8_SMEM_FLOATS(512, 32, 32) * sizeof(float);
    if (smemBp != smemB) { printf("!! bench copy and shipped arm disagree on smem\n"); return 1; }
#endif
    printf("# arm A smem %zu B (shipped px4 fp8mma, BQ32/BKV16)   arm B smem %zu B (BQ32/BKV32)\n",
           smemA, smemB);
    printf("# fp8 peak (in-tree rtx-05) = 503.8 TFLOP/s ; bf16/fp16 peak = 209.5\n\n");

    if (!strcmp(mode, "layout") || !strcmp(mode, "all")) { run_layout(); printf("\n"); }

    /* ---- P.V phase microbench ---- */
    if (!strcmp(mode, "pv") || !strcmp(mode, "all")) {
        printf("== P.V PHASE per 32 kv rows, 1 block/SM, %d blocks (V already in smem as e4m3)\n", P);
        float* sink; CHK(cudaMalloc(&sink, 4));
        const int it = 20000;
#define PVSMEM(VPAD) ((size_t)32 * (HD + (VPAD)) + (size_t)32 * (HD + PADB) * 2 + 32 * 4)
#define PVSET(M, VPAD, NJ)                                                                          \
    CHK(cudaFuncSetAttribute(k_pvphase<M, VPAD, NJ>,                                                \
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)PVSMEM(VPAD)))
        PVSET(0, 32, 16); PVSET(1, 32, 16); PVSET(2, 32, 16); PVSET(1, 16, 16); PVSET(1, 32, 32);
        double t0 = timeit([&] { k_pvphase<0><<<P, 256, PVSMEM(32)>>>(sink, it); }, 1, 5);
        double t1 = timeit([&] { k_pvphase<1><<<P, 256, PVSMEM(32)>>>(sink, it); }, 1, 5);
        double t2 = timeit([&] { k_pvphase<2><<<P, 256, PVSMEM(32)>>>(sink, it); }, 1, 5);
        double t3 = timeit([&] { k_pvphase<1, 16><<<P, 256, PVSMEM(16)>>>(sink, it); }, 1, 5);
        double t4 = timeit([&] { k_pvphase<1, 32, 32><<<P, 256, PVSMEM(32)>>>(sink, it); }, 1, 5);
        printf("  %-46s %8.2f ns / 32-kv tile\n", "A: dequant->fp16 + 32x ldmatrix.b16 + 32x f16 mma",
               t0 * 1e6 / it);
        printf("  %-46s %8.2f ns / 32-kv tile   %.2fx\n",
               "B: 8x ldmatrix.m16n16.trans.b8 + 16x e4m3 mma", t1 * 1e6 / it, t0 / t1);
        printf("  %-46s %8.2f ns / 32-kv tile   %.2fx\n", "Bg: 8xLDS.8 gather + 16x e4m3 mma",
               t2 * 1e6 / it, t0 / t2);
        printf("  %-46s %8.2f ns / 32-kv tile   %.2fx\n", "B with Vs8 row pad 16 (2-way vs 4-way)",
               t3 * 1e6 / it, t0 / t3);
        /* NJ_PV=32 is the BQ=64 accumulator shape. It does 2x the hd per warp, so the honest
         * reading is the ratio to 2*t1 (>1 means the extra 64 accumulator registers cost), NOT
         * the ratio to t1. The register count from -Xptxas -v is the load-bearing number. */
        printf("  %-46s %8.2f ns / 2x the hd per warp  %.3fx vs 2x B\n",
               "B at NJ_PV=32 (the BQ=64 acc shape)", t4 * 1e6 / it, t4 / (2 * t1));
        cudaFree(sink);
        printf("\n");
    }

    /* ---- numerics ---- */
    if (!strcmp(mode, "num") || !strcmp(mode, "all")) {
        printf("== NUMERICS vs an f32 reference (same e4m3 K/V, same per-row scales)\n");
        const unsigned sq = 256, skv = 1024, nh = 4, nkv = 1;
        Data d = make(sq, skv, nh, nkv, skv - sq);
        const float sc = 1.0f / sqrtf((float)HD);
        k_ref<<<dim3(sq, nh), 256>>>(d.ref, d.Q, d.K, d.V, d.ksc, d.vsc, sq, skv, nh, nkv,
                                     d.q_pos0, sc);
        CHK(cudaDeviceSynchronize());
        CHK(cudaFuncSetAttribute(k_armA, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemA));
        CHK(cudaFuncSetAttribute(k_armB<0, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)smemB));
        CHK(cudaFuncSetAttribute(k_armB<0, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)smemB));
        CHK(cudaMemset(d.O, 0, (size_t)sq * nh * HD * 2));
        k_armA<<<P, 256, smemA>>>(d.O, d.Q, (const __nv_bfloat16*)d.K, (const __nv_bfloat16*)d.V,
                                  d.ksc, d.vsc, sq, skv, nh, nkv, d.q_pos0, skv, skv - 1, sc);
        CHK(cudaDeviceSynchronize()); score("A  shipped px4 fp8mma (fp16 P.V)", d);
        CHK(cudaMemset(d.O, 0, (size_t)sq * nh * HD * 2));
        k_armB<0, 1><<<P, 256, smemB>>>(d.O, d.Q, (const __nv_bfloat16*)d.K,
                                        (const __nv_bfloat16*)d.V, d.ksc, d.vsc, sq, skv, nh, nkv,
                                        d.q_pos0, skv, skv - 1, sc);
        CHK(cudaDeviceSynchronize()); score("Bg e4m3 P.V, LDS.8 gather (oracle)", d);
        CHK(cudaMemset(d.O, 0, (size_t)sq * nh * HD * 2));
        k_armB<0, 0><<<P, 256, smemB>>>(d.O, d.Q, (const __nv_bfloat16*)d.K,
                                        (const __nv_bfloat16*)d.V, d.ksc, d.vsc, sq, skv, nh, nkv,
                                        d.q_pos0, skv, skv - 1, sc);
        CHK(cudaDeviceSynchronize()); score("B  e4m3 P.V, ldmatrix.trans.b8", d);
#if PLOW_NV_FA_FP8PV
        CHK(cudaFuncSetAttribute(k_armBp, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)smemBp));
        CHK(cudaMemset(d.O, 0, (size_t)sq * nh * HD * 2));
        k_armBp<<<P, 256, smemBp>>>(d.O, d.Q, (const __nv_bfloat16*)d.K, (const __nv_bfloat16*)d.V,
                                    d.ksc, d.vsc, sq, skv, nh, nkv, d.q_pos0, skv, skv - 1, sc);
        CHK(cudaDeviceSynchronize()); score("Bp SHIPPED d_flash_prefill_px8", d);
#endif
        freeData(d);
        printf("  (B must match Bg to the bit-ish; if it does not, the ldmatrix fragment map is\n"
               "   wrong and every B timing below is measuring the wrong instruction sequence.)\n\n");
    }

    /* ---- full-kernel A/B at real shapes ---- */
    if (!strcmp(mode, "full") || !strcmp(mode, "all")) {
        printf("== FULL KERNEL, trailing 8k chunk, nh=16 nkv=1 (Gemma-4-12B full layer), grid=%d\n",
               P);
        printf("%8s %9s %10s %10s %8s %10s %10s %8s\n", "seq_kv", "A ms", "A ns/tile", "A TFLOP/s",
               "B ms", "B ns/tile", "B TFLOP/s", "B/A");
        const unsigned sq = 8192, nh = 16, nkv = 1;
        CHK(cudaFuncSetAttribute(k_armA, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemA));
        CHK(cudaFuncSetAttribute(k_armB<0, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)smemB));
        const unsigned kvs[] = {8192, 32768, 131072};
        for (unsigned skv : kvs) {
            Data d = make(sq, skv, nh, nkv, skv - sq);
            const float sc = 1.0f / sqrtf((float)HD);
            const int iters = (skv > 65536) ? 3 : 10;
            double ta = timeit([&] {
                k_armA<<<P, 256, smemA>>>(d.O, d.Q, (const __nv_bfloat16*)d.K,
                                          (const __nv_bfloat16*)d.V, d.ksc, d.vsc, sq, skv, nh,
                                          nkv, d.q_pos0, skv, skv - 1, sc);
            }, 2, iters);
            double tb = timeit([&] {
                k_armB<0, 0><<<P, 256, smemB>>>(d.O, d.Q, (const __nv_bfloat16*)d.K,
                                                (const __nv_bfloat16*)d.V, d.ksc, d.vsc, sq, skv,
                                                nh, nkv, d.q_pos0, skv, skv - 1, sc);
            }, 2, iters);
            /* FLOPs are tiling-independent: 2 mmas x 2 flop x (query,kv) pairs x HD */
            double pairs = 0;
            for (unsigned q = 0; q < sq; q++) {
                long qabs = (long)d.q_pos0 + q;
                pairs += (double)((qabs + 1 < (long)skv) ? qabs + 1 : (long)skv);
            }
            const double fl = 2.0 /*QK,PV*/ * 2.0 * pairs * nh * HD;
#if PLOW_NV_FA_FP8PV
            CHK(cudaFuncSetAttribute(k_armBp, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     (int)smemBp));
            double tp = timeit([&] {
                k_armBp<<<P, 256, smemBp>>>(d.O, d.Q, (const __nv_bfloat16*)d.K,
                                            (const __nv_bfloat16*)d.V, d.ksc, d.vsc, sq, skv, nh,
                                            nkv, d.q_pos0, skv, skv - 1, sc);
            }, 2, iters);
            printf("%8s %9s %10s %10s %8.3f %10.1f %10.1f %8.3f  <- Bp SHIPPED px8\n", "", "", "",
                   "", tp, tp * 1e6 / (tile_count(sq, skv, nh, 32, 32) / P),
                   fl / (tp * 1e-3) / 1e12, tp / ta);
#endif
            printf("%8u %9.3f %10.1f %10.1f %8.3f %10.1f %10.1f %8.3f\n", skv, ta,
                   ta * 1e6 / (tile_count(sq, skv, nh, 32, 16) / P),
                   fl / (ta * 1e-3) / 1e12, tb,
                   tb * 1e6 / (tile_count(sq, skv, nh, 32, 32) / P),
                   fl / (tb * 1e-3) / 1e12, tb / ta);
            freeData(d);
        }
        printf("  (ns/tile is per BLOCK per KV tile of that arm's own BKV; flat across seq_kv\n"
               "   means the kernel is NOT KV-traffic bound and BQ=64 cannot help.)\n\n");
    }

    /* ---- arm-B phase ablation ---- */
    if (!strcmp(mode, "abl") || !strcmp(mode, "all")) {
        printf("== ABLATION (arm B, seq_kv=32768): delta vs full = that phase's exposed cost\n");
        const unsigned sq = 8192, skv = 32768, nh = 16, nkv = 1;
        Data d = make(sq, skv, nh, nkv, skv - sq);
        const float sc = 1.0f / sqrtf((float)HD);
#define ABLRUN(BITS, NAME)                                                                          \
    do {                                                                                            \
        CHK(cudaFuncSetAttribute(k_armB<BITS, 0>,                                                    \
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemB));         \
        double t = timeit([&] {                                                                     \
            k_armB<BITS, 0><<<P, 256, smemB>>>(d.O, d.Q, (const __nv_bfloat16*)d.K,                 \
                                               (const __nv_bfloat16*)d.V, d.ksc, d.vsc, sq, skv,    \
                                               nh, nkv, d.q_pos0, skv, skv - 1, sc);                \
        }, 2, 10);                                                                                  \
        if (BITS == 0) base = t;                                                                    \
        printf("  %-34s %8.3f ms   delta %7.3f ms  (%5.1f%%)\n", NAME, t, base - t,                 \
               100.0 * (base - t) / base);                                                          \
    } while (0)
        double base = 1;
        ABLRUN(0, "full (control)");
        ABLRUN(1, "- QK mma");
        ABLRUN(2, "- softmax (incl. P quant+rescale)");
        ABLRUN(4, "- P.V mma");
        ABLRUN(8, "- cp.async issue (no gmem)");
        ABLRUN(15, "- everything (loop+barrier floor)");
#undef ABLRUN
        freeData(d);
    }
    return 0;
}
