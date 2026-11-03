/* moe_decode_gemma26b_bench.cu — time the Gemma-4-26B-A4B DECODE MoE ops in isolation.
 *
 * WHY: served 26B decode runs at TPOT 48.8 ms against vLLM's 5.03 ms, and static analysis
 * could not identify the cost. Two hypotheses were killed by comparing against the 12B
 * packet, which serves correctly on the same binaries:
 *   - "the MoE expert GEMV is bandwidth-starved": 26B streams 7.64 GB/token vs 12B's
 *     23.8 GB and is still 3.8x slower per stage, so it is not bytes.
 *   - "the tiny 1-block ops starve the grid": 12B has the SAME distribution
 *     (55 % of stages on <=16 of 132 blocks vs 26B's 59 %) and is faster anyway.
 * The decode step is ONE cooperative launch, so there is no per-stage timing and no
 * profiler attach (see plans note on compute-sanitizer / ncu). Measuring the ops
 * standalone is the only way to get a per-op number.
 *
 * Reports per-op time and the effective bandwidth against the bytes each op must read,
 * at the real geometry (H 2816, I_moe 704, E 128, top-k 8) and over the decode rungs.
 *
 * Build: nvcc -gencode arch=compute_90a,code=sm_90a -O3 -DPLOW_NV_GEMMA=1 \
 *        -DPLOW_NV_HOPPER=1 -DPLOW_NV_GEMV_RB=1 -DPLOW_MOE_DOWN_LANESPLIT=1 \
 *        -I runtime/common -I runtime/nvidia runtime/tests/moe_decode_gemma26b_bench.cu
 * Row-invariance check of the served router: add the recipe's -DPLOW_NV_ROUTER_V8=1
 * -DPLOW_NV_TOPK_V4=1 -DPLOW_NV_ROUTER_V9=1 (and -DPLOW_NV_ROUTER_ROW_INVARIANT=1), run with
 * MOE_CHECK_ONLY=1.
 */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#include "sm120_common.cuh"
#include "op_norm.cuh"
#include "op_elementwise.cuh"
#include "op_gemm.cuh"
#include "dev_isa.h"
#include "op_moe.cuh"

typedef __nv_bfloat16 bf16;
#define CK(x) do { cudaError_t e_=(x); if (e_!=cudaSuccess){ \
    printf("FAIL %s at %d: %s\n",#x,__LINE__,cudaGetErrorString(e_)); exit(2);} } while(0)

static uint32_t rs = 0x2468ace0u;
static float rnd() { rs ^= rs<<13; rs ^= rs>>17; rs ^= rs<<5; return (float)((int32_t)rs)/2147483648.0f; }

extern __shared__ bf16 g_arena[];

__global__ void k_score(float* score, const bf16* resid, const bf16* proj, const bf16* scale,
                        unsigned H, unsigned n_exp, float root, float eps, unsigned nrow) {
    d_moe_router_gemma_score(score, resid, proj, scale, H, n_exp, root, eps,
                             blockIdx.x, gridDim.x, nrow);
}

/* The emit default (ScoreFast); the recipe's PLOW_GEMMA_MOE_ROUTER_EXACT=1 selects k_score. */
__global__ void k_score_fast(float* score, const bf16* resid, const bf16* proj, const bf16* scale,
                             unsigned H, unsigned n_exp, float root, float eps, unsigned nrow) {
    d_moe_router_gemma_score_fast(score, resid, proj, scale, H, n_exp, root, eps,
                                  blockIdx.x, gridDim.x, nrow, (float*)g_arena);
}

__global__ void k_topk(unsigned char* table, const float* score, const bf16* pes,
                       unsigned n_exp, unsigned k, unsigned nrow) {
    d_moe_router_gemma_topk(table, score, pes, n_exp, k, blockIdx.x, gridDim.x, nrow,
                            (float*)g_arena);
}

__global__ void k_glu(bf16* fu, const bf16* x, const unsigned char* table,
                      const unsigned long long* ewt, unsigned k, unsigned I_moe, unsigned H,
                      unsigned n_exp, unsigned nrow) {
    d_moe_expert_glu_gemma(fu, x, table, ewt, k, I_moe, H, n_exp, blockIdx.x, gridDim.x, nrow,
                           g_arena);
}

/* The op the bf16 packet actually emits (fused norm + GLU). xn == nullptr takes its scalar
 * batch body; a scratch takes the staged vector body. */
__global__ void k_glu_norm(bf16* fu, const bf16* resid, const bf16* gamma,
                           const unsigned char* table, const unsigned long long* ewt, unsigned k,
                           unsigned I_moe, unsigned H, unsigned n_exp, unsigned nrow, bf16* xn) {
    d_moe_expert_glu_norm_gemma(fu, resid, gamma, table, ewt, k, I_moe, H, n_exp, 1e-6f,
                                blockIdx.x, gridDim.x, nrow, (float*)g_arena, xn);
}

__global__ void k_down(float* part, const bf16* fu, const unsigned char* table,
                       const unsigned long long* ewt, unsigned k, unsigned H, unsigned I_moe,
                       unsigned n_exp, unsigned nrow) {
    d_moe_expert_down_gemma(part, fu, table, ewt, k, H, I_moe, n_exp, blockIdx.x, gridDim.x, nrow,
                            (float*)g_arena);
}

template <class F>
static double time_ms(F&& launch, int iters) {
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    launch(); CK(cudaDeviceSynchronize());          /* warm */
    CK(cudaEventRecord(a));
    for (int i = 0; i < iters; i++) launch();
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms = 0; CK(cudaEventElapsedTime(&ms, a, b));
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    return (double)ms / iters;
}

int main(int argc, char** argv) {
    const unsigned H = 2816, I_moe = 704, n_exp = 128, k = 8;
    const unsigned grid = 132, iters = argc > 1 ? (unsigned)atoi(argv[1]) : 50u;
    const unsigned LAYERS = 30;

    /* Expert weights at FULL per-layer size: one layer's fused tensors. The served packet
     * holds 30 of these; a decode token reads top-k of each. */
    const size_t gu_elems = (size_t)n_exp * 2u * I_moe * H;
    const size_t dn_elems = (size_t)n_exp * H * I_moe;
    printf("geometry H=%u I_moe=%u E=%u k=%u | one layer: gate_up %.2f GB, down %.2f GB\n",
           H, I_moe, n_exp, k, gu_elems*2.0/1e9, dn_elems*2.0/1e9);

    bf16 *d_gu=nullptr,*d_dn=nullptr;
    CK(cudaMalloc(&d_gu, gu_elems*sizeof(bf16)));
    CK(cudaMalloc(&d_dn, dn_elems*sizeof(bf16)));
    CK(cudaMemset(d_gu, 0x3c, gu_elems*sizeof(bf16)));
    CK(cudaMemset(d_dn, 0x3c, dn_elems*sizeof(bf16)));

    std::vector<unsigned long long> hewt(n_exp*2);
    for (unsigned e=0;e<n_exp;e++){
        hewt[e*2+0]=(unsigned long long)(size_t)(d_gu+(size_t)e*2u*I_moe*H);
        hewt[e*2+1]=(unsigned long long)(size_t)(d_dn+(size_t)e*H*I_moe);
    }
    unsigned long long* d_ewt=nullptr;
    CK(cudaMalloc(&d_ewt,hewt.size()*sizeof(unsigned long long)));
    CK(cudaMemcpy(d_ewt,hewt.data(),hewt.size()*sizeof(unsigned long long),cudaMemcpyHostToDevice));

    std::vector<bf16> hproj((size_t)n_exp*H), hscale(H), hpes(n_exp);
    for(auto&v:hproj) v=__float2bfloat16(rnd()*0.05f);
    for(auto&v:hscale) v=__float2bfloat16(1.f);
    for(auto&v:hpes) v=__float2bfloat16(1.f);
    bf16 *d_proj=nullptr,*d_scale=nullptr,*d_pes=nullptr;
    CK(cudaMalloc(&d_proj,hproj.size()*sizeof(bf16)));
    CK(cudaMemcpy(d_proj,hproj.data(),hproj.size()*sizeof(bf16),cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_scale,hscale.size()*sizeof(bf16)));
    CK(cudaMemcpy(d_scale,hscale.data(),hscale.size()*sizeof(bf16),cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_pes,hpes.size()*sizeof(bf16)));
    CK(cudaMemcpy(d_pes,hpes.data(),hpes.size()*sizeof(bf16),cudaMemcpyHostToDevice));

    const size_t arena_bytes = (size_t)PGM_ARENA_BF16*sizeof(bf16);
    /* Every one of these bodies indexes the shared arena; launching with zero dynamic
     * smem makes the first store an illegal access. Raise the cap on all four. */
    CK(cudaFuncSetAttribute(k_glu,  cudaFuncAttributeMaxDynamicSharedMemorySize,(int)arena_bytes));
    CK(cudaFuncSetAttribute(k_down, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)arena_bytes));
    CK(cudaFuncSetAttribute(k_score,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)arena_bytes));
    CK(cudaFuncSetAttribute(k_topk, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)arena_bytes));

    /* Row invariance of the served router (ScoreFast + top-k at the emitter's CTA counts): row 0
     * must score and route to the same bits at every decode rung whatever its companions are.
     * Non-unit scale so the h2 association is visible. Exit 1 on any difference. */
    {
        CK(cudaFuncSetAttribute(k_score_fast,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)arena_bytes));
        std::vector<bf16> hs(H);
        for (auto& v : hs) v = __float2bfloat16(1.0f + rnd() * 0.5f);
        bf16* d_s = nullptr; CK(cudaMalloc(&d_s, H * sizeof(bf16)));
        CK(cudaMemcpy(d_s, hs.data(), H * sizeof(bf16), cudaMemcpyHostToDevice));
        const unsigned BMAX = 32;
        std::vector<bf16> hx((size_t)BMAX * H);
        for (auto& v : hx) v = __float2bfloat16(rnd() * 2.0f);
        bf16* d_x = nullptr; CK(cudaMalloc(&d_x, hx.size() * sizeof(bf16)));
        float* d_sc = nullptr; CK(cudaMalloc(&d_sc, (size_t)BMAX * n_exp * sizeof(float)));
        unsigned char* d_tb = nullptr; CK(cudaMalloc(&d_tb, (size_t)BMAX * k * 8));
        const float root = 1.0f / sqrtf((float)H);
        std::vector<float> s1(n_exp), sb(n_exp);
        std::vector<unsigned char> t1(k * 8), tb(k * 8);
        bool ok = true;
        for (unsigned B : {1u, 2u, 4u, 8u, 16u, 32u}) {
            for (unsigned r = 1; r < B; r++)  /* fresh companions per rung */
                for (unsigned h = 0; h < H; h++) hx[(size_t)r * H + h] = __float2bfloat16(rnd() * 2.0f);
            CK(cudaMemcpy(d_x, hx.data(), (size_t)B * H * sizeof(bf16), cudaMemcpyHostToDevice));
            k_score_fast<<<(B * n_exp + 7u) / 8u, 256, arena_bytes>>>(d_sc, d_x, d_proj, d_s, H, n_exp, root, 1e-6f, B);
            k_topk<<<B, 256, arena_bytes>>>(d_tb, d_sc, d_pes, n_exp, k, B);
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(sb.data(), d_sc, n_exp * sizeof(float), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(tb.data(), d_tb, k * 8, cudaMemcpyDeviceToHost));
            if (B == 1) { s1 = sb; t1 = tb; continue; }
            const bool same_s = memcmp(s1.data(), sb.data(), n_exp * sizeof(float)) == 0;
            const bool same_t = memcmp(t1.data(), tb.data(), k * 8) == 0;
            printf("ROW_INVARIANT B=%u scores %s table %s\n", B, same_s ? "same" : "DIFF", same_t ? "same" : "DIFF");
            ok = ok && same_s && same_t;
        }
        printf("ROW_INVARIANT %s\n", ok ? "PASS" : "FAIL");
        cudaFree(d_s); cudaFree(d_x); cudaFree(d_sc); cudaFree(d_tb);
        if (!ok || getenv("MOE_CHECK_ONLY")) return ok ? 0 : 1;
    }

    printf("\n%6s | %10s %10s %10s %10s | %9s %9s\n",
           "B","score ms","topk ms","glu ms","down ms","MoE/tok","GB/s glu");
    for (unsigned B : {1u,2u,4u,8u,16u}) {
        std::vector<bf16> hx((size_t)B*H);
        for(auto&v:hx) v=__float2bfloat16(rnd()*0.05f);
        bf16* d_x=nullptr; CK(cudaMalloc(&d_x,hx.size()*sizeof(bf16)));
        CK(cudaMemcpy(d_x,hx.data(),hx.size()*sizeof(bf16),cudaMemcpyHostToDevice));
        float* d_score=nullptr; CK(cudaMalloc(&d_score,(size_t)B*n_exp*sizeof(float)));
        unsigned char* d_tab=nullptr; CK(cudaMalloc(&d_tab,(size_t)B*k*8));
        bf16* d_fu=nullptr; CK(cudaMalloc(&d_fu,(size_t)B*k*I_moe*sizeof(bf16)));
        float* d_part=nullptr; CK(cudaMalloc(&d_part,(size_t)B*k*H*sizeof(float)));
        const float root = 1.0f/sqrtf((float)H);

        /* score CTA count mirrors the packet: 8 experts per CTA over the (row,expert) pairs. */
        const unsigned gscore = (B*n_exp + 7u)/8u;
        double t_sc = time_ms([&]{ k_score<<<gscore,256,arena_bytes>>>(d_score,d_x,d_proj,d_scale,H,n_exp,root,1e-6f,B); }, iters);
        CK(cudaFuncSetAttribute(k_score_fast,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)arena_bytes));
        double t_sf = time_ms([&]{ k_score_fast<<<gscore,256,arena_bytes>>>(d_score,d_x,d_proj,d_scale,H,n_exp,root,1e-6f,B); }, iters);
        printf("         score_fast %.4f ms (exact %.4f)\n", t_sf, t_sc);
        double t_tk = time_ms([&]{ k_topk<<<1,256,arena_bytes>>>(d_tab,d_score,d_pes,n_exp,k,B); }, iters);
        CK(cudaDeviceSynchronize());
        double t_gl = time_ms([&]{ k_glu<<<grid,256,arena_bytes>>>(d_fu,d_x,d_tab,d_ewt,k,I_moe,H,n_exp,B); }, iters);
        /* served op: fused norm+GLU, scalar batch body vs the moe.xn2-staged vector body. */
        bf16 *d_gam = nullptr, *d_xn = nullptr, *d_fu2 = nullptr;
        {
            std::vector<bf16> gam(H);
            for (auto& v : gam) v = __float2bfloat16(1.0f + rnd() * 0.1f);
            CK(cudaMalloc(&d_gam, H * sizeof(bf16)));
            CK(cudaMemcpy(d_gam, gam.data(), H * sizeof(bf16), cudaMemcpyHostToDevice));
            CK(cudaMalloc(&d_xn, (size_t)B * H * sizeof(bf16)));
            CK(cudaMalloc(&d_fu2, (size_t)B * k * I_moe * sizeof(bf16)));
        }
        CK(cudaFuncSetAttribute(k_glu_norm,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)arena_bytes));
        double t_gn0 = time_ms([&]{ k_glu_norm<<<grid,256,arena_bytes>>>(d_fu,d_x,d_gam,d_tab,d_ewt,k,I_moe,H,n_exp,B,nullptr); }, iters);
        double t_gn1 = time_ms([&]{ k_glu_norm<<<grid,256,arena_bytes>>>(d_fu2,d_x,d_gam,d_tab,d_ewt,k,I_moe,H,n_exp,B,d_xn); }, iters);
        {
            std::vector<bf16> a((size_t)B * k * I_moe), c(a.size());
            CK(cudaMemcpy(a.data(), d_fu, a.size() * sizeof(bf16), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(c.data(), d_fu2, c.size() * sizeof(bf16), cudaMemcpyDeviceToHost));
            double num = 0, den = 0;
            for (size_t i = 0; i < a.size(); i++) {
                const double x = __bfloat162float(a[i]), y = __bfloat162float(c[i]);
                num += (x - y) * (x - y); den += x * x;
            }
            printf("         glu_norm scalar %.4f ms -> staged-vector %.4f ms   relL2 %.2e\n",
                   t_gn0, t_gn1, den > 0 ? sqrt(num / den) : 0.0);
        }
        cudaFree(d_gam); cudaFree(d_xn); cudaFree(d_fu2);
        double t_dn = time_ms([&]{ k_down<<<grid,256,arena_bytes>>>(d_part,d_fu,d_tab,d_ewt,k,H,I_moe,n_exp,B); }, iters);
        CK(cudaDeviceSynchronize());
        if (const char* dump = getenv("MOE_DN_DUMP")) { /* numeric compare across two builds */
            std::vector<float> hp((size_t)B * k * H);
            CK(cudaMemcpy(hp.data(), d_part, hp.size() * sizeof(float), cudaMemcpyDeviceToHost));
            char path[512];
            snprintf(path, sizeof path, "%s.B%u", dump, B);
            FILE* fp = fopen(path, "wb"); fwrite(hp.data(), sizeof(float), hp.size(), fp); fclose(fp);
        }

        /* Bytes the GLU must read at this batch: union of experts picked, x 2I x H x 2B.
         * Upper bound = min(B*k, n_exp) distinct experts. */
        const double experts = (double)(B*k < n_exp ? B*k : n_exp);
        const double glu_bytes = experts * 2.0 * I_moe * H * 2.0;
        const double per_tok_ms = (t_sc+t_tk+t_gl+t_dn) * LAYERS / (double)B;
        printf("%6u | %10.4f %10.4f %10.4f %10.4f | %8.2f%s %9.0f\n",
               B, t_sc, t_tk, t_gl, t_dn, per_tok_ms, "ms", glu_bytes/(t_gl*1e-3)/1e9);
        cudaFree(d_x); cudaFree(d_score); cudaFree(d_tab); cudaFree(d_fu); cudaFree(d_part);
    }
    printf("\nper-tok = (score+topk+glu+down) x %u layers / B  — the MoE share of one decode token\n", LAYERS);
    return 0;
}
