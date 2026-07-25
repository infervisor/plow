// px4_fa_ablate.cu — PX-4 phase-cost attribution for the hd512 FULL-layer flash prefill.
//
// ncu is unavailable on this box (RmProfilingAdminOnly=1 inside the container), so this harness
// attributes the per-KV-tile cost of the PIPE=1 d_flash_prefill<512,32,16> analytically: it embeds
// a byte-faithful copy of the T5 cp.async kernel (op_attention.cuh PLOW_NV_FA_PIPE=1, VBUF=1) with
// compile-time ABLATE switches that null out ONE phase at a time while keeping every barrier,
// commit and wait_group in place, so the delta vs the full kernel is that phase's cost alone.
//   bit 0: skip QK mma       (ldmatrix+mma of S; Ss stays stale)
//   bit 1: skip softmax      (warp reductions + exp2; Ps stays stale)
//   bit 2: skip P.V mma      (ldmatrix+mma of O)
//   bit 3: skip cp.async ISSUE (no gmem traffic at all; empty commit keeps group counts right)
// Timing convention identical to fa_pv_bench.cu (grid 188 / 256 threads, zeroed data, cudaEvent).
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "op_attention.cuh"

#define CHK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA ERR %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

template <int HD, int BQ, int BKV, int ABL>
__device__ void d_fa_ablate(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
                            const __nv_bfloat16* __restrict__ V, __nv_bfloat16* __restrict__ O,
                            unsigned seq_q, unsigned seq_kv, unsigned n_head, unsigned n_kv_head,
                            unsigned q_pos0, float scale, unsigned slice, unsigned nblk, float* lds) {
    constexpr int PAD = FA_PRE_PAD;
    constexpr int WQK_N = 2;
    constexpr int WQK_M = BQ / 16;
    constexpr int WN = BKV / WQK_N;
    constexpr int NJ = WN / 8;
    constexpr int KSTEPS = HD / 16;
    constexpr int WPV_M = BQ / 16;
    constexpr int WPV_N = (int)PLOW_NV_WARPS / WPV_M;
    constexpr int HDW = HD / WPV_N;
    constexpr int NJ_PV = HDW / 8;
    constexpr int KSTEPS_PV = BKV / 16;
    constexpr int RPW_S = BQ / (int)PLOW_NV_WARPS;
    constexpr int HCH = HD / 8;

    float* Ss = lds;
    float* m_arr = Ss + BQ * BKV;
    float* l_arr = m_arr + BQ;
    float* corr_arr = l_arr + BQ;
    __nv_bfloat16* Qs = (__nv_bfloat16*)(corr_arr + BQ);
    __nv_bfloat16* Ks = Qs + BQ * (HD + PAD);
    __nv_bfloat16* Vs = Ks + BKV * (HD + PAD);
    __nv_bfloat16* Ps = Vs + BKV * (HD + PAD);

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const unsigned gqa = n_head / n_kv_head;
    const unsigned n_qt = (seq_q + BQ - 1) / BQ;
    const unsigned n_work = n_qt * n_head;
    const float lscale = FA_SCALE(scale);
    const int qk_wm = warp / WQK_N, qk_wn = warp % WQK_N;
    const bool qk_active = warp < WQK_M * WQK_N;
    const int pv_wm = warp / WPV_N, pv_wn = warp % WPV_N;

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
        for (int r = tid; r < BQ; r += (int)PLOW_NV_THREADS) {
            m_arr[r] = FA_NEG_INF;
            l_arr[r] = 0.0f;
        }

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

            if (!(ABL & 1) && qk_active) {
                float acc[NJ][4];
#pragma unroll
                for (int nj = 0; nj < NJ; nj++)
#pragma unroll
                    for (int e = 0; e < 4; e++) acc[nj][e] = 0.f;
#pragma unroll
                for (int kf = 0; kf < KSTEPS; kf++) {
                    unsigned af[4];
                    fa_ldmatrix_x4(af, &Qs[(qk_wm * 16 + (lane % 16)) * (HD + PAD) +
                                          kf * 16 + (lane / 16) * 8]);
                    unsigned bf[NJ][2];
#pragma unroll
                    for (int nj = 0; nj < NJ; nj++) {
                        const int n = qk_wn * WN + nj * 8 + (lane & 7);
                        const int kcol = kf * 16 + ((lane >> 3) & 1) * 8;
                        fa_ldmatrix_x2(bf[nj], &Ks[n * (HD + PAD) + kcol]);
                    }
#pragma unroll
                    for (int nj = 0; nj < NJ; nj++) fa_mma(acc[nj], af, bf[nj], acc[nj]);
                }
#pragma unroll
                for (int nj = 0; nj < NJ; nj++)
#pragma unroll
                    for (int e = 0; e < 4; e++) {
                        int qr = qk_wm * 16 + (lane / 4) + (e / 2) * 8;
                        int kc = qk_wn * WN + nj * 8 + (lane % 4) * 2 + (e % 2);
                        Ss[qr * BKV + kc] = acc[nj][e] * lscale;
                    }
            }
            __syncthreads();

            if (t + 1 < nt) stageK(kv0 + BKV);
            else fa_cp_commit();

            const unsigned rmax = (hi - kv0 < (unsigned)BKV) ? (hi - kv0) : (unsigned)BKV;

            if (!(ABL & 2)) {
#pragma unroll
                for (int rr = 0; rr < RPW_S; rr++) {
                    const int row = warp * RPW_S + rr;
                    const int qabs = (int)(q_pos0 + q0 + row);
                    float sv = FA_NEG_INF;
                    bool active = false;
                    if ((unsigned)lane < rmax) {
                        const int kv = (int)kv0 + lane;
                        bool masked = (kv > qabs);
                        if (!masked) { sv = Ss[row * BKV + lane]; active = true; }
                    }
                    const float rowmax = warp_max32(sv);
                    const float m_old = m_arr[row];
                    const float m_new = fmaxf(m_old, rowmax);
                    const float corr = (m_old == FA_NEG_INF) ? 0.0f : FA_EXP(m_old - m_new);
                    const float p = (active && m_new != FA_NEG_INF) ? FA_EXP(sv - m_new) : 0.0f;
                    if (lane < BKV) Ps[row * (BKV + PAD) + lane] = __float2bfloat16(p);
                    const float rowsum = warp_sum32(p);
                    if (lane == 0) {
                        l_arr[row] = l_arr[row] * corr + rowsum;
                        m_arr[row] = m_new;
                        corr_arr[row] = corr;
                    }
                }
            }
            fa_cp_wait<1>();
            __syncthreads();

            if (!(ABL & 4)) {
                const float c_lo = corr_arr[pv_wm * 16 + (lane >> 2)];
                const float c_hi = corr_arr[pv_wm * 16 + (lane >> 2) + 8];
#pragma unroll
                for (int nj = 0; nj < NJ_PV; nj++) {
                    oacc[nj][0] *= c_lo;
                    oacc[nj][1] *= c_lo;
                    oacc[nj][2] *= c_hi;
                    oacc[nj][3] *= c_hi;
                }
#pragma unroll
                for (int kf = 0; kf < KSTEPS_PV; kf++) {
                    unsigned af[4];
                    fa_ldmatrix_x4(af, &Ps[(pv_wm * 16 + (lane % 16)) * (BKV + PAD) +
                                          kf * 16 + (lane / 16) * 8]);
#pragma unroll
                    for (int nj = 0; nj < NJ_PV; nj++) {
                        unsigned bf[2];
                        fa_ldmatrix_x2_trans(bf, &Vs[(kf * 16 + (lane % 16)) * (HD + PAD) +
                                                     pv_wn * HDW + nj * 8]);
                        fa_mma(oacc[nj], af, bf, oacc[nj]);
                    }
                }
            }
            __syncthreads();
        }

        /* Epilogue (kept: negligible, once per work item). */
#pragma unroll
        for (int nj = 0; nj < NJ_PV; nj++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const unsigned qrow = (unsigned)(pv_wm * 16 + (lane >> 2) + (e >> 1) * 8);
                const unsigned qabs_row = q0 + qrow;
                if (qabs_row >= seq_q) continue;
                const int hd = pv_wn * HDW + nj * 8 + (lane & 3) * 2 + (e & 1);
                const float lv = l_arr[qrow];
                const float inv = (lv > 0.0f) ? (1.0f / lv) : 0.0f;
                O[(size_t)(qabs_row * n_head + h) * HD + hd] = __float2bfloat16(oacc[nj][e] * inv);
            }
    }
}

template <int HD, int BQ, int BKV, int ABL>
__global__ void k_abl(__nv_bfloat16* O, const __nv_bfloat16* Q, const __nv_bfloat16* K,
                      const __nv_bfloat16* V, unsigned sq, unsigned skv, unsigned nh, unsigned nkv,
                      unsigned qpos0, float scale) {
    extern __shared__ float sm[];
    d_fa_ablate<HD, BQ, BKV, ABL>(Q, K, V, O, sq, skv, nh, nkv, qpos0, scale, blockIdx.x,
                                  gridDim.x, sm);
}

template <int ABL>
static double bench(const char* label, unsigned seq_kv, int iters) {
    constexpr int HD = 512, BQ = 32, BKV = 16;
    const unsigned nh = 16, nkv = 1, seq_q = 8192;
    const unsigned q_pos0 = (seq_kv > seq_q) ? (seq_kv - seq_q) : 0u;
    size_t nQ = (size_t)seq_q * nh * HD, nKV = (size_t)nkv * seq_kv * HD;
    __nv_bfloat16 *dQ, *dK, *dV, *dO;
    CHK(cudaMalloc(&dQ, nQ * 2)); CHK(cudaMalloc(&dK, nKV * 2));
    CHK(cudaMalloc(&dV, nKV * 2)); CHK(cudaMalloc(&dO, nQ * 2));
    CHK(cudaMemset(dQ, 0, nQ * 2)); CHK(cudaMemset(dK, 0, nKV * 2));
    CHK(cudaMemset(dV, 0, nKV * 2)); CHK(cudaMemset(dO, 0, nQ * 2));
    const size_t smem = (size_t)FA_PRE_SMEM_FLOATS(HD, BQ, BKV) * sizeof(float);
    CHK(cudaFuncSetAttribute(k_abl<HD, BQ, BKV, ABL>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    for (int i = 0; i < 2; i++)
        k_abl<HD, BQ, BKV, ABL><<<188, 256, smem>>>(dO, dQ, dK, dV, seq_q, seq_kv, nh, nkv, q_pos0, 1.0f);
    CHK(cudaDeviceSynchronize());
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < iters; i++)
        k_abl<HD, BQ, BKV, ABL><<<188, 256, smem>>>(dO, dQ, dK, dV, seq_q, seq_kv, nh, nkv, q_pos0, 1.0f);
    cudaEventRecord(b); CHK(cudaEventSynchronize(b));
    float ms = 0; cudaEventElapsedTime(&ms, a, b); ms /= iters;
    printf("  %-44s %9.3f ms\n", label, ms);
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms;
}

int main(int argc, char** argv) {
    unsigned skv = (argc > 1) ? (unsigned)atoi(argv[1]) : 8192;
    int iters = (argc > 2) ? atoi(argv[2]) : (skv > 65536 ? 5 : 20);
    cudaDeviceProp p; CHK(cudaGetDeviceProperties(&p, 0));
    printf("device: %s  SMs=%d  seq_kv=%u\n", p.name, p.multiProcessorCount, skv);
    bench<0>("FULL kernel (control, ~= fa_pv_bench)", skv, iters);
    bench<1>("- QK mma skipped", skv, iters);
    bench<2>("- softmax skipped", skv, iters);
    bench<4>("- P.V mma skipped", skv, iters);
    bench<8>("- cp.async issue skipped (no gmem traffic)", skv, iters);
    bench<3>("- QK+softmax skipped", skv, iters);
    bench<7>("- QK+softmax+PV skipped (staging only)", skv, iters);
    bench<15>("- everything skipped (loop+barrier floor)", skv, iters);
    return 0;
}
