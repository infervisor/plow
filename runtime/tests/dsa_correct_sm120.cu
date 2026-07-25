/* dsa_correct_sm120.cu — NUMERIC ORACLE for the sm_120 GLM DSA indexer arms (P3).
 *
 * Drives the three DSA op bodies the interp wires (op_dsa.cuh d_index_score_sm120 +
 * d_index_select_sm120, op_norm.cuh d_layernorm_bias) against a self-contained f32 CPU golden that
 * reproduces runtime/tests/mla_ref.rs indexer_selfcheck VERBATIM (same splitmix64 seeding, same
 * bf16 rounding, same dsa_pack_key <<24 select), plus an inline f32 LayerNorm ref. These are the
 * SAME device functions interp_sm120.cu inlines into the megakernel, called with the identical
 * operand order as the INDEX_SCORE / INDEX_SELECT / LAYERNORM dispatch arms.
 *
 *   SCORE   : device score vs CPU index_score           -> rel_rms
 *   SELECT  : device top-k SET vs CPU brute-force top-k  -> set-match (like the AMD glm52 harness)
 *   LAYERNORM: device vs inline f32 (x-μ)·rsqrt(var+eps)·γ+β -> rel_rms
 *
 * Build (needs a GPU at run time):
 *   env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin nvcc -std=c++17 -arch=sm_120a -O3 \
 *     -I runtime/common -I runtime/nvidia runtime/tests/dsa_correct_sm120.cu -o dsa_correct
 */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <algorithm>

#include "op_norm.cuh" /* d_layernorm_bias (+ pulls op_attention.cuh / sm120_common.cuh) */
#include "op_dsa.cuh"  /* d_index_score_sm120 / d_index_select_sm120 */

typedef __nv_bfloat16 bf16;
#define CK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){printf("CUDA ERR %s: %s\n",#x,cudaGetErrorString(e_));exit(2);} } while(0)

/* ---- CPU golden bit-math, byte-identical to mla_ref.rs ---- */
static uint16_t f2bf(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    uint32_t r = u + (0x7fffu + ((u >> 16) & 1u));
    return (uint16_t)(r >> 16);
}
static float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
static uint64_t mix(uint64_t z) {
    z += 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
static float rnd(uint64_t seed, float amp) {
    uint64_t u = mix(seed);
    float unit = (float)(u >> 40) / (float)(1u << 24);
    return (unit * 2.0f - 1.0f) * amp;
}
/* ---- test __global__ wrappers: exactly the interp dispatch call, slice=blockIdx nblk=gridDim ---- */
__global__ void k_index_score(float* Score, const bf16* Qidx, const bf16* Kidx, const bf16* W,
                              const int* kvlen, unsigned nb, unsigned kvs, float scale) {
    extern __shared__ float arena[];
    d_index_score_sm120<128, 32>(Score, Qidx, Kidx, W, kvlen, nb, kvs, scale, blockIdx.x, gridDim.x,
                                 arena);
}
__global__ void k_index_select(int* idx, const float* Score, unsigned len, unsigned top_k,
                               unsigned* gHist, unsigned* gCtl) {
    d_index_select_sm120<true>(idx, /*n_sel*/ nullptr, Score, len, top_k, gHist, gCtl, blockIdx.x,
                               gridDim.x);
}
__global__ void k_layernorm(bf16* out, const bf16* x, const bf16* g, const bf16* b, unsigned rows,
                            unsigned feat, unsigned out_row0, float eps) {
    extern __shared__ float part[];
    d_layernorm_bias(out, x, g, b, rows, feat, out_row0, eps, blockIdx.x, gridDim.x, part);
}

static int g_fail = 0;

static void rms(const char* what, const std::vector<float>& got, const std::vector<float>& want,
                double gate) {
    double se = 0, sw = 0, md = 0, mw = 0;
    for (size_t i = 0; i < got.size(); i++) {
        double d = fabs((double)got[i] - want[i]);
        se += d * d; sw += (double)want[i] * want[i];
        md = fmax(md, d); mw = fmax(mw, fabs(want[i]));
    }
    double rel_rms = sqrt(se / got.size()) / (sqrt(sw / got.size()) + 1e-12);
    double rel_max = md / (mw + 1e-12);
    bool ok = rel_rms < gate;
    printf("  %-40s %s  (rel_rms %.5f  rel_max %.4f)\n", what, ok ? "PASS" : "FAIL", rel_rms, rel_max);
    if (!ok) g_fail = 1;
}

int main() {
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    printf("dev0: %s  SMs=%d\n\n", prop.name, prop.multiProcessorCount);

    const unsigned HI = 32, DI = 128;
    const float scale = powf((float)DI, -0.5f) * powf((float)HI, -0.5f);
    const int nb_score = std::min(188, prop.multiProcessorCount); /* score: grid-strided slabs */
    const unsigned nwg = 32;                                      /* select: co-resident CUs */

    struct Case { unsigned ctx, top_k; };
    Case cases[] = {{4096, 2048}, {32768, 2048}, {2000, 512}, {131072, 2048}};

    printf("DSA INDEX SCORE + SELECT (device vs mla_ref indexer golden):\n");
    for (Case c : cases) {
        const unsigned ctx = c.ctx, top_k = c.top_k;
        const uint64_t seed = ((uint64_t)ctx << 20) ^ top_k;

        /* --- inputs (bf16), seeded exactly as indexer_selfcheck --- */
        std::vector<uint16_t> qb(HI * DI), kb((size_t)ctx * DI), wb(HI);
        std::vector<float> qf(HI * DI), kf((size_t)ctx * DI), wf(HI);
        for (unsigned i = 0; i < HI * DI; i++) { qb[i] = f2bf(rnd(seed ^ 0x11 ^ i, 0.1f)); qf[i] = bf2f(qb[i]); }
        for (unsigned i = 0; i < HI; i++)      { wb[i] = f2bf(rnd(seed ^ 0x22 ^ i, 1.0f)); wf[i] = bf2f(wb[i]); }
        for (size_t i = 0; i < (size_t)ctx * DI; i++) { kb[i] = f2bf(rnd(seed ^ 0x33 ^ i, 0.1f)); kf[i] = bf2f(kb[i]); }

        /* --- CPU golden score + brute-force top-k set --- */
        std::vector<float> gold_sc(ctx);
        for (unsigned t = 0; t < ctx; t++) {
            float s = 0.0f;
            for (unsigned h = 0; h < HI; h++) {
                float d = 0.0f;
                for (unsigned i = 0; i < DI; i++) d += qf[h * DI + i] * kf[(size_t)t * DI + i];
                s += wf[h] * (d > 0.0f ? d : 0.0f);
            }
            gold_sc[t] = s * scale;
        }
        std::vector<unsigned> order(ctx);
        for (unsigned t = 0; t < ctx; t++) order[t] = t;
        std::sort(order.begin(), order.end(), [&](unsigned a, unsigned b) {
            return gold_sc[a] != gold_sc[b] ? gold_sc[a] > gold_sc[b] : a < b;
        });
        std::vector<char> want_sel(ctx, 0);
        for (unsigned r = 0; r < top_k; r++) want_sel[order[r]] = 1;

        /* --- device SCORE --- */
        bf16 *dQ, *dK, *dW; float* dSc; int L = (int)ctx; int* dLen;
        CK(cudaMalloc(&dQ, qb.size() * 2));  CK(cudaMemcpy(dQ, qb.data(), qb.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dK, kb.size() * 2));  CK(cudaMemcpy(dK, kb.data(), kb.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dW, wb.size() * 2));  CK(cudaMemcpy(dW, wb.data(), wb.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dLen, 4)); CK(cudaMemcpy(dLen, &L, 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dSc, (size_t)ctx * 4));
        const size_t smem = (size_t)DSA_SCORE_SMEM_FLOATS(128, 32, DSA_SCORE_TILE_N) * sizeof(float);
        CK(cudaFuncSetAttribute(k_index_score, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        k_index_score<<<nb_score, 256, smem>>>(dSc, dQ, dK, dW, dLen, 1, ctx, scale);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        std::vector<float> dev_sc(ctx);
        CK(cudaMemcpy(dev_sc.data(), dSc, (size_t)ctx * 4, cudaMemcpyDeviceToHost));
        char lbl[96];
        snprintf(lbl, sizeof(lbl), "score  ctx=%-7u", ctx);
        rms(lbl, dev_sc, gold_sc, 5e-3);

        /* --- device SELECT (cooperative launch, nwg co-resident CUs) --- */
        int* dIdx; unsigned *dHist, *dCtl;
        CK(cudaMalloc(&dIdx, (size_t)top_k * 4));
        CK(cudaMalloc(&dHist, (size_t)DSA_SEL_NPASS * DSA_SEL_NB * 4)); CK(cudaMemset(dHist, 0, (size_t)DSA_SEL_NPASS * DSA_SEL_NB * 4));
        CK(cudaMalloc(&dCtl, 3 * 4)); CK(cudaMemset(dCtl, 0, 3 * 4));
        void* args[] = {&dIdx, &dSc, (void*)&ctx, (void*)&top_k, &dHist, &dCtl};
        CK(cudaLaunchCooperativeKernel((void*)k_index_select, dim3(nwg), dim3(256), args, 0, 0));
        CK(cudaDeviceSynchronize());
        std::vector<int> dev_idx(top_k);
        CK(cudaMemcpy(dev_idx.data(), dIdx, (size_t)top_k * 4, cudaMemcpyDeviceToHost));
        unsigned match = 0; bool oob = false;
        for (unsigned r = 0; r < top_k; r++) {
            int p = dev_idx[r];
            if (p < 0 || (unsigned)p >= ctx) { oob = true; continue; }
            if (want_sel[p]) match++;
        }
        bool sok = (match == top_k) && !oob;
        printf("  select ctx=%-7u %s  (set-match %u/%u%s)\n", ctx, sok ? "PASS" : "FAIL", match,
               top_k, oob ? " OOB!" : "");
        if (!sok) g_fail = 1;

        cudaFree(dQ); cudaFree(dK); cudaFree(dW); cudaFree(dLen); cudaFree(dSc);
        cudaFree(dIdx); cudaFree(dHist); cudaFree(dCtl);
    }

    /* --- LayerNorm+bias vs inline f32 ref --- */
    printf("\nDSA LAYERNORM+bias (device vs inline f32 ref):\n");
    for (unsigned feat : {128u, 256u}) {
        const unsigned rows = 1;
        const float eps = 1e-6f;
        std::vector<uint16_t> xb(feat), gb(feat), bb(feat);
        std::vector<float> xf(feat), gf(feat), bf(feat);
        for (unsigned i = 0; i < feat; i++) {
            xb[i] = f2bf(rnd(0xA1u ^ i, 1.0f)); xf[i] = bf2f(xb[i]);
            gb[i] = f2bf(rnd(0xA2u ^ i, 0.5f)); gf[i] = bf2f(gb[i]);
            bb[i] = f2bf(rnd(0xA3u ^ i, 0.5f)); bf[i] = bf2f(bb[i]);
        }
        double mean = 0, msq = 0;
        for (unsigned i = 0; i < feat; i++) { mean += xf[i]; msq += (double)xf[i] * xf[i]; }
        mean /= feat; msq /= feat;
        double inv = 1.0 / sqrt(msq - mean * mean + eps);
        std::vector<float> gold(feat);
        for (unsigned i = 0; i < feat; i++)
            gold[i] = bf2f(f2bf((float)(((double)xf[i] - mean) * inv * gf[i] + bf[i])));

        bf16 *dX, *dG, *dB, *dO;
        CK(cudaMalloc(&dX, feat * 2)); CK(cudaMemcpy(dX, xb.data(), feat * 2, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dG, feat * 2)); CK(cudaMemcpy(dG, gb.data(), feat * 2, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dB, feat * 2)); CK(cudaMemcpy(dB, bb.data(), feat * 2, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dO, feat * 2));
        k_layernorm<<<1, 256, 64 * sizeof(float)>>>(dO, dX, dG, dB, rows, feat, 0, eps);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        std::vector<uint16_t> ob(feat);
        CK(cudaMemcpy(ob.data(), dO, feat * 2, cudaMemcpyDeviceToHost));
        std::vector<float> dev(feat);
        for (unsigned i = 0; i < feat; i++) dev[i] = bf2f(ob[i]);
        char lbl[64]; snprintf(lbl, sizeof(lbl), "layernorm feat=%u", feat);
        rms(lbl, dev, gold, 5e-3);
        cudaFree(dX); cudaFree(dG); cudaFree(dB); cudaFree(dO);
    }

    printf("\nRESULT: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail;
}
