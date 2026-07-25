// px4_fp8mma_ablate.cu — phase-cost attribution for the PX-4 hd512 FULL-layer flash prefill
// (beat-fp8-mma-prefill step 0).
//
// The committed px4_fa_ablate.cu ablates the PRE-px4 T5 kernel (smem softmax + Ps + 4-warp QK);
// its budget (softmax 36% / QK 21%) is the map px4 was BUILT FROM, not the map of px4 itself.
// This harness embeds a byte-faithful copy of the CURRENT d_flash_prefill_px4<512,32,16,false>
// (PIPE=1, non-TMA, bf16 arm) with compile-time ABL bits that null ONE phase at a time while
// keeping every barrier/commit/wait in place, so the delta vs the full kernel is that phase's
// exposed cost alone.
//   bit 0 (1):  skip QK mma        (ldmatrix+mma+score store; SsA/SsB stay stale)
//   bit 1 (2):  skip softmax       (reg reductions + EX2 + P pack + oacc rescale; af_pv = 0)
//   bit 2 (4):  skip P.V mma       (ldmatrix.trans + mma of O)
//   bit 3 (8):  skip cp.async ISSUE (no gmem traffic; empty commits keep group counts)
//   bit 4 (16): skip ONLY the oacc corr-rescale (the 64-fmul/lane per-tile multiply inside softmax)
//   bit 5 (32): FA_EXP -> identity (keeps the dependency chain, removes the MUFU.EX2s)
// Timing convention identical to px4_fa_ablate.cu (grid 188 x 256, zeroed data, cudaEvent).
// Work shape mirrors a real trailing 8k chunk: seq_q=8192, q_pos0=seq_kv-8192, NH=16, NKV=2.
//
// Build: env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin nvcc -std=c++17 -arch=sm_120a -O3 \
//          -I runtime/common -I runtime/nvidia -include cstdint \
//          runtime/nvidia/experiments/px4_fp8mma_ablate.cu -o /tmp/px4_abl
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "op_attention.cuh"

#define CHK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA ERR %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

template <int HD, int BQ, int BKV, int ABL>
__device__ void d_px4_abl(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
                          const __nv_bfloat16* __restrict__ V, __nv_bfloat16* __restrict__ O,
                          unsigned seq_q, unsigned seq_kv, unsigned n_head, unsigned n_kv_head,
                          unsigned q_pos0, float scale, unsigned slice, unsigned nblk, float* lds) {
    static_assert(HD == 512 && BQ == 32 && BKV == 16, "px4 arm tiling");
    constexpr int PAD = FA_PRE_PAD;
    constexpr int KSTEPS_H = HD / 2 / 16;
    constexpr int WPV_N = 4;
    constexpr int HDW = HD / WPV_N;
    constexpr int NJ_PV = HDW / 8;
    constexpr int HCH = HD / 8;

    float* SsA = lds + 4;
    float* SsB = SsA + BQ * BKV;
    __nv_bfloat16* Qs = (__nv_bfloat16*)(SsB + BQ * BKV);
    __nv_bfloat16* Ks = Qs + BQ * (HD + PAD);
    __nv_bfloat16* Vs = Ks + BKV * (HD + PAD);

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const unsigned gqa = n_head / n_kv_head;
    const unsigned n_qt = (seq_q + BQ - 1) / BQ;
    const unsigned n_work = n_qt * n_head;
    const float lscale = FA_SCALE(scale);
    const int qk_kh = warp >> 2, qk_wm = (warp >> 1) & 1, qk_wn = warp & 1;
    const int pv_wm = warp >> 2, pv_wn = warp & 3;
    const int r0 = pv_wm * 16 + (lane >> 2);
    const int c0 = (lane & 3) * 2;

    for (unsigned w = slice; w < n_work; w += nblk) {
        const unsigned h = w % n_head;
        const unsigned qt = w / n_head;
        const unsigned q0 = qt * BQ;
        const unsigned hkv = h / gqa;
        const unsigned lo = 0, hi = seq_kv;

        const __nv_bfloat16* Qh = Q + (size_t)q0 * n_head * HD + (size_t)h * HD;
        const __nv_bfloat16* Kb = K + (size_t)hkv * seq_kv * HD;
        const __nv_bfloat16* Vb = V + (size_t)hkv * seq_kv * HD;

        __syncthreads();
        for (int idx = tid; idx < BQ * HD; idx += (int)PLOW_NV_THREADS) {
            int r = idx / HD, c = idx % HD;
            __nv_bfloat16 v = __float2bfloat16(0.f);
            if (q0 + r < seq_q) v = Qh[(size_t)r * n_head * HD + c];
            Qs[r * (HD + PAD) + c] = v;
        }

        float oacc[NJ_PV][4];
#pragma unroll
        for (int nj = 0; nj < NJ_PV; nj++)
#pragma unroll
            for (int e = 0; e < 4; e++) oacc[nj][e] = 0.0f;
        float m_reg[2] = {FA_NEG_INF, FA_NEG_INF}, l_reg[2] = {0.0f, 0.0f};

        const int qabs_max = (int)(q_pos0 + q0 + BQ - 1);
        long cap = (long)hi - 1;
        if ((long)qabs_max < cap) cap = (long)qabs_max;
        const int nt = (cap >= (long)lo) ? (int)((cap - (long)lo) / BKV) + 1 : 0;

        auto stageK = [&](unsigned kv0) {
            if (!(ABL & 8))
                for (int L = tid; L < BKV * HCH; L += (int)PLOW_NV_THREADS) {
                    int r = L / HCH, c8 = (L % HCH) * 8;
                    unsigned kv = kv0 + (unsigned)r;
                    bool in = (kv < hi);
                    const __nv_bfloat16* g = in ? Kb + (size_t)kv * HD + c8 : Kb;
                    fa_cp_async_cg16(&Ks[r * (HD + PAD) + c8], g, in ? 16 : 0);
                }
            fa_cp_commit();
        };
        auto stageV = [&](unsigned kv0) {
            if (!(ABL & 8))
                for (int L = tid; L < BKV * HCH; L += (int)PLOW_NV_THREADS) {
                    int r = L / HCH, c8 = (L % HCH) * 8;
                    unsigned kv = kv0 + (unsigned)r;
                    bool in = (kv < hi);
                    const __nv_bfloat16* g = in ? Vb + (size_t)kv * HD + c8 : Vb;
                    fa_cp_async_cg16(&Vs[r * (HD + PAD) + c8], g, in ? 16 : 0);
                }
            fa_cp_commit();
        };

        __syncthreads();
        if (nt > 0) stageK(lo);

        for (int t = 0; t < nt; t++) {
            const unsigned kv0 = lo + (unsigned)t * BKV;
            stageV(kv0);
            fa_cp_wait<1>();
            __syncthreads();

            if (!(ABL & 1)) {
                float acc[4] = {0.f, 0.f, 0.f, 0.f};
                const int khoff = qk_kh * (HD / 2);
#pragma unroll
                for (int kf = 0; kf < KSTEPS_H; kf++) {
                    unsigned af[4];
                    fa_ldmatrix_x4(af, &Qs[(qk_wm * 16 + (lane % 16)) * (HD + PAD) + khoff +
                                          kf * 16 + (lane / 16) * 8]);
                    unsigned bf[2];
                    {
                        const int n = qk_wn * 8 + (lane & 7);
                        const int kcol = khoff + kf * 16 + ((lane >> 3) & 1) * 8;
                        fa_ldmatrix_x2(bf, &Ks[n * (HD + PAD) + kcol]);
                    }
                    fa_mma(acc, af, bf, acc);
                }
                float* Sdst = qk_kh ? SsB : SsA;
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    int qr = qk_wm * 16 + (lane / 4) + (e / 2) * 8;
                    int kc = qk_wn * 8 + (lane % 4) * 2 + (e % 2);
                    Sdst[qr * BKV + kc] = acc[e] * lscale;
                }
            }
            __syncthreads();

            if (t + 1 < nt) stageK(kv0 + BKV);
            else fa_cp_commit();

            const unsigned rmax = (hi - kv0 < (unsigned)BKV) ? (hi - kv0) : (unsigned)BKV;

            unsigned af_pv[4] = {0u, 0u, 0u, 0u};
            if (!(ABL & 2)) {
                float p[2][4];
                float corr[2];
#pragma unroll
                for (int j = 0; j < 2; j++) {
                    const int row = r0 + j * 8;
                    const int qabs = (int)(q_pos0 + q0 + row);
                    float s[4], mx = FA_NEG_INF;
#pragma unroll
                    for (int ci = 0; ci < 4; ci++) {
                        const int col = c0 + (ci & 1) + (ci >> 1) * 8;
                        const int kv = (int)kv0 + col;
                        bool masked = ((unsigned)col >= rmax) || (kv > qabs);
                        s[ci] = masked ? FA_NEG_INF : SsA[row * BKV + col] + SsB[row * BKV + col];
                        mx = fmaxf(mx, s[ci]);
                    }
                    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
                    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
                    const float m_new = fmaxf(m_reg[j], mx);
                    corr[j] = (m_reg[j] == FA_NEG_INF)
                                  ? 0.0f
                                  : ((ABL & 32) ? (m_reg[j] - m_new) : FA_EXP(m_reg[j] - m_new));
                    float lsum = 0.0f;
#pragma unroll
                    for (int ci = 0; ci < 4; ci++) {
                        p[j][ci] = (s[ci] == FA_NEG_INF || m_new == FA_NEG_INF)
                                       ? 0.0f
                                       : ((ABL & 32) ? (s[ci] - m_new) : FA_EXP(s[ci] - m_new));
                        lsum += p[j][ci];
                    }
                    lsum += __shfl_xor_sync(0xffffffffu, lsum, 1);
                    lsum += __shfl_xor_sync(0xffffffffu, lsum, 2);
                    l_reg[j] = l_reg[j] * corr[j] + lsum;
                    m_reg[j] = m_new;
                }
                __nv_bfloat162 h2;
                h2 = __floats2bfloat162_rn(p[0][0], p[0][1]); af_pv[0] = *(unsigned*)&h2;
                h2 = __floats2bfloat162_rn(p[1][0], p[1][1]); af_pv[1] = *(unsigned*)&h2;
                h2 = __floats2bfloat162_rn(p[0][2], p[0][3]); af_pv[2] = *(unsigned*)&h2;
                h2 = __floats2bfloat162_rn(p[1][2], p[1][3]); af_pv[3] = *(unsigned*)&h2;
                if (!(ABL & 16)) {
#pragma unroll
                    for (int nj = 0; nj < NJ_PV; nj++) {
                        oacc[nj][0] *= corr[0];
                        oacc[nj][1] *= corr[0];
                        oacc[nj][2] *= corr[1];
                        oacc[nj][3] *= corr[1];
                    }
                }
            }

            fa_cp_wait<1>();
            __syncthreads();

            if (!(ABL & 4)) {
#pragma unroll
                for (int nj = 0; nj < NJ_PV; nj++) {
                    unsigned bf[2];
                    fa_ldmatrix_x2_trans(bf, &Vs[(lane % 16) * (HD + PAD) + pv_wn * HDW + nj * 8]);
                    fa_mma(oacc[nj], af_pv, bf, oacc[nj]);
                }
            }
            __syncthreads();
        }

        /* Epilogue (kept: once per work item, negligible). nsplit==1 normalise-in-place form. */
#pragma unroll
        for (int nj = 0; nj < NJ_PV; nj++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const unsigned qrow = (unsigned)(r0 + (e >> 1) * 8);
                const unsigned qabs_row = q0 + qrow;
                if (qabs_row >= seq_q) continue;
                const int hd = pv_wn * HDW + nj * 8 + (lane & 3) * 2 + (e & 1);
                const float lv = l_reg[e >> 1];
                const float inv = (lv > 0.0f) ? (1.0f / lv) : 0.0f;
                O[(size_t)(qabs_row * n_head + h) * HD + hd] = __float2bfloat16(oacc[nj][e] * inv);
            }
    }
}

template <int HD, int BQ, int BKV, int ABL>
__global__ void __launch_bounds__(256, 1)
k_abl(__nv_bfloat16* O, const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
      unsigned sq, unsigned skv, unsigned nh, unsigned nkv, unsigned qpos0, float scale) {
    extern __shared__ float sm[];
    d_px4_abl<HD, BQ, BKV, ABL>(Q, K, V, O, sq, skv, nh, nkv, qpos0, scale, blockIdx.x, gridDim.x,
                                sm);
}

template <int ABL>
static double bench(const char* label, unsigned seq_kv, int iters, double full_ms) {
    constexpr int HD = 512, BQ = 32, BKV = 16;
    const unsigned nh = 16, nkv = 2, seq_q = 8192;
    const unsigned q_pos0 = (seq_kv > seq_q) ? (seq_kv - seq_q) : 0u;
    size_t nQ = (size_t)seq_q * nh * HD, nKV = (size_t)nkv * seq_kv * HD;
    __nv_bfloat16 *dQ, *dK, *dV, *dO;
    CHK(cudaMalloc(&dQ, nQ * 2)); CHK(cudaMalloc(&dK, nKV * 2));
    CHK(cudaMalloc(&dV, nKV * 2)); CHK(cudaMalloc(&dO, nQ * 2));
    CHK(cudaMemset(dQ, 0, nQ * 2)); CHK(cudaMemset(dK, 0, nKV * 2));
    CHK(cudaMemset(dV, 0, nKV * 2)); CHK(cudaMemset(dO, 0, nQ * 2));
    const size_t smem = (size_t)FA_PX4_SMEM_FLOATS(HD, BQ, BKV) * sizeof(float);
    CHK(cudaFuncSetAttribute(k_abl<HD, BQ, BKV, ABL>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    for (int i = 0; i < 2; i++)
        k_abl<HD, BQ, BKV, ABL><<<188, 256, smem>>>(dO, dQ, dK, dV, seq_q, seq_kv, nh, nkv, q_pos0,
                                                    1.0f);
    CHK(cudaDeviceSynchronize());
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < iters; i++)
        k_abl<HD, BQ, BKV, ABL><<<188, 256, smem>>>(dO, dQ, dK, dV, seq_q, seq_kv, nh, nkv, q_pos0,
                                                    1.0f);
    cudaEventRecord(b); CHK(cudaEventSynchronize(b));
    float ms = 0; cudaEventElapsedTime(&ms, a, b); ms /= iters;
    if (full_ms > 0)
        printf("  %-46s %9.3f ms   delta %8.3f ms  (%5.1f%% of full)\n", label, ms, full_ms - ms,
               100.0 * (full_ms - ms) / full_ms);
    else
        printf("  %-46s %9.3f ms\n", label, ms);
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms;
}

int main(int argc, char** argv) {
    unsigned skv = (argc > 1) ? (unsigned)atoi(argv[1]) : 32768;
    int iters = (argc > 2) ? atoi(argv[2]) : (skv > 65536 ? 5 : 20);
    cudaDeviceProp p; CHK(cudaGetDeviceProperties(&p, 0));
    /* per-tile normalization: total KV tiles across all (qt,h) work items */
    const unsigned seq_q = 8192, nh = 16, BQ = 32, BKV = 16;
    const unsigned q_pos0 = (skv > seq_q) ? (skv - seq_q) : 0u;
    double tiles = 0;
    for (unsigned qt = 0; qt < seq_q / BQ; qt++) {
        long cap = (long)skv - 1;
        long qm = (long)q_pos0 + qt * BQ + BQ - 1;
        if (qm < cap) cap = qm;
        if (cap >= 0) tiles += (double)(cap / BKV + 1);
    }
    tiles *= nh;
    printf("device: %s  SMs=%d  seq_kv=%u  seq_q=8192 (trailing chunk)  tiles=%.0f\n", p.name,
           p.multiProcessorCount, skv, tiles);
    double full = bench<0>("FULL px4 kernel (control)", skv, iters, 0);
    printf("  per-tile (188 SMs): %.1f ns\n", full * 1e6 / (tiles / 188.0));
    bench<1>("- QK mma skipped", skv, iters, full);
    bench<2>("- softmax skipped (incl. rescale)", skv, iters, full);
    bench<16>("- oacc corr-rescale ONLY skipped", skv, iters, full);
    bench<32>("- EX2 -> identity (MUFU removed)", skv, iters, full);
    bench<4>("- P.V mma skipped", skv, iters, full);
    bench<8>("- cp.async issue skipped (no gmem)", skv, iters, full);
    bench<3>("- QK+softmax skipped", skv, iters, full);
    bench<7>("- QK+softmax+PV skipped (staging only)", skv, iters, full);
    bench<15>("- everything skipped (loop+barrier floor)", skv, iters, full);
    /* compute-side decomposition: with the memory stream removed (bit 8), the run is pure
     * compute; skipping one compute phase on top isolates its share of the COMPUTE side. */
    double comp = bench<8|0>("COMPUTE side (mem skipped, control)", skv, iters, 0);
    bench<8|1>("  compute - QK mma", skv, iters, comp);
    bench<8|2>("  compute - softmax", skv, iters, comp);
    bench<8|4>("  compute - P.V mma", skv, iters, comp);
    bench<8|1|4>("  compute - QK - P.V (softmax only)", skv, iters, comp);
    return 0;
}
