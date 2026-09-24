/* moe_group_gemma26b_repro.cu — ISOLATED sm_90a repro for the Gemma-4-26B-A4B grouped-MoE
 * PREFILL chain: op 74 (ALIGN) -> op 75 (GROUP_GLU) -> op 76 (GROUP_DOWN).
 *
 * WHY: serving the real 26B packet faults with CUDA_ERROR_ILLEGAL_ADDRESS on the FIRST prefill
 * chunk (39-token prompt, 128-row bucket), with every segment role turned off. The existing
 * runtime/tests/moe_group_bench.cu covers this chain, but it targets sm_120a, which takes the
 * mma.sync arm of op_moe.cuh. An H100 takes the OTHER arm -- the wgmma fork in op_moe_sm90.cuh,
 * included only under PLOW_NV_HOPPER -- so the Hopper bodies have no coverage at this geometry.
 * This reproduces the exact chain, at the exact shapes, without the 47 GB checkpoint.
 *
 * Geometry is the real one: H 2816, I_moe 704, E 128, top-k 8. Note 704 = 5.5 * 128, so the
 * N tiling has a HALF-WIDTH final tile -- the only MoE geometry in the tree with that property.
 *
 * Build: nvcc -arch=sm_90a -O3 -DPLOW_NV_GEMMA=1 -DPLOW_NV_HOPPER=1 -I runtime/common \
 *          -I runtime/nvidia runtime/tests/moe_group_gemma26b_repro.cu -o moerepro
 */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <cmath>

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

__global__ void k_align(int* meta, const unsigned char* table, unsigned* row_token,
                        unsigned* row_partidx, float* row_gate, unsigned T, unsigned n_exp,
                        unsigned k) {
    d_moe_align_gemma_pf(meta, table, row_token, row_partidx, row_gate, T, n_exp, k, blockIdx.x);
}

extern __shared__ bf16 g_arena[];

__global__ void k_glu(bf16* fu, const bf16* xn2, const unsigned long long* ewt, const int* meta,
                      const unsigned* row_token, unsigned I_moe, unsigned H, unsigned n_exp,
                      unsigned act) {
    d_moe_group_glu_gemma_pf(fu, xn2, ewt, meta, row_token, I_moe, H, n_exp, act,
                             blockIdx.x, gridDim.x, g_arena);
}

__global__ void k_down(float* part, const bf16* fu, const unsigned long long* ewt, const int* meta,
                       const unsigned* row_partidx, const float* row_gate, unsigned H,
                       unsigned I_moe, unsigned n_exp) {
    d_moe_group_down_gemma_pf(part, fu, ewt, meta, row_partidx, row_gate, H, I_moe, n_exp,
                              blockIdx.x, gridDim.x, g_arena);
}

/* op 73: the T-token router. Runs FIRST and, in a real bucket, over rows of
 * uninitialised padding — so it is fed deliberately hostile activations below. */
__global__ void k_router(unsigned char* table, const bf16* resid, const bf16* proj,
                         const bf16* scale, const bf16* pes, unsigned H, unsigned n_exp,
                         unsigned k, unsigned T, float root, float eps) {
    d_moe_router_gemma_pf(table, resid, proj, scale, pes, H, n_exp, k, T, root, eps,
                          blockIdx.x, gridDim.x, (float*)g_arena);
}

/* op 77: T-row combine + sandwich norm. */
__global__ void k_comb(bf16* out, const float* part, const bf16* h1, const bf16* gamma,
                       unsigned H, unsigned k, unsigned T, float eps) {
    d_moe_combine_norm_gemma_pf(out, part, h1, gamma, H, k, T, eps, blockIdx.x, gridDim.x,
                                (float*)g_arena);
}

int main(int argc, char** argv) {
    const unsigned T      = argc > 1 ? (unsigned)atoi(argv[1]) : 128u;
    const unsigned H      = 2816u;
    const unsigned I_moe  = 704u;
    const unsigned n_exp  = 128u;
    const unsigned k      = 8u;
    const unsigned grid   = argc > 2 ? (unsigned)atoi(argv[2]) : 132u;
    /* Exactly the emitter's bound (devgen lib.rs): moe_rows*top_k + n_exp*PGM_BM. */
    const size_t total_pad = (size_t)T * k + (size_t)n_exp * 128u;

    printf("T=%u H=%u I_moe=%u E=%u k=%u grid=%u total_pad=%zu  (I_moe/BN = %.2f tiles)\n",
           T, H, I_moe, n_exp, k, grid, total_pad, (double)I_moe / 128.0);

    /* Routing table: [T*k] of {u32 eid, f32 gate}.
     * dist 0 = even spread (every expert gets the same count).
     * dist 1 = ALL slots to expert 0: 127 empty experts, one segment many tiles deep.
     * dist 2 = two hot experts + a long tail, the shape a real prompt produces when most
     *          of the bucket is uninitialised padding and routing degenerates.
     * The even case is NOT representative: a real 128-row bucket holding a 39-token prompt
     * runs the router over ~89 rows of garbage. */
    const unsigned dist = argc > 3 ? (unsigned)atoi(argv[3]) : 0u;
    const size_t nslot = (size_t)T * k;
    std::vector<unsigned char> tab(nslot * 8);
    for (size_t s = 0; s < nslot; s++) {
        unsigned eid;
        if (dist == 1) eid = 0u;
        else if (dist == 2) eid = (s % 16u < 14u) ? (unsigned)(s & 1u) : (unsigned)(2u + (s % 61u));
        else eid = (unsigned)((s * 37u + (s >> 3)) % n_exp);
        float gate = 0.125f;
        memcpy(&tab[s*8+0], &eid, 4);
        memcpy(&tab[s*8+4], &gate, 4);
    }
    /* dist 4 = REPLAY a routing table dumped from the model (block_run check --dump-tensors
     * act.moe.table): real routing is ~30 hot experts + a long tail + ~33 empty, which none of
     * the synthetic shapes above has. */
    if (dist == 4) {
        const char* path = getenv("MOE_RT_LOAD");
        FILE* f = path ? fopen(path, "rb") : nullptr;
        if (!f || fread(tab.data(), 1, tab.size(), f) != tab.size()) {
            fprintf(stderr, "dist 4 needs MOE_RT_LOAD=<table dump of >= T*k*8 bytes>\n");
            return 2;
        }
        fclose(f);
    }
    printf("routing dist=%u\n", dist);

    std::vector<float> hx((size_t)T * H);
    for (auto& v : hx) v = rnd() * 0.05f;
    std::vector<bf16> xn2(hx.size());
    for (size_t i = 0; i < hx.size(); i++) xn2[i] = __float2bfloat16(hx[i]);

    /* Fused expert tensors, exactly as the checkpoint stores them. */
    const size_t gu_elems = (size_t)n_exp * 2u * I_moe * H;   /* [E, 2I, H] */
    const size_t dn_elems = (size_t)n_exp * H * I_moe;        /* [E, H, I]  */
    printf("expert weights: gate_up %.2f GB, down %.2f GB\n",
           gu_elems * 2.0 / 1e9, dn_elems * 2.0 / 1e9);

    bf16 *d_gu = nullptr, *d_dn = nullptr, *d_x = nullptr, *d_fu = nullptr;
    CK(cudaMalloc(&d_gu, gu_elems * sizeof(bf16)));
    CK(cudaMalloc(&d_dn, dn_elems * sizeof(bf16)));
    /* Expert-distinct, row-distinct weights: 4 random base matrices dealt round-robin, so a
     * wrong-expert or wrong-n-tile read changes `part` instead of hiding in a constant fill. */
    {
        const size_t gu1 = (size_t)2u * I_moe * H, dn1 = (size_t)H * I_moe;
        std::vector<bf16> bgu(4 * gu1), bdn(4 * dn1);
        for (auto& v : bgu) v = __float2bfloat16(rnd() * 0.02f);
        for (auto& v : bdn) v = __float2bfloat16(rnd() * 0.02f);
        for (unsigned e = 0; e < n_exp; e++) {
            const unsigned b = (e * 7u + (e >> 3)) & 3u;
            CK(cudaMemcpy(d_gu + (size_t)e * gu1, bgu.data() + b * gu1, gu1 * sizeof(bf16), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_dn + (size_t)e * dn1, bdn.data() + b * dn1, dn1 * sizeof(bf16), cudaMemcpyHostToDevice));
        }
    }
    CK(cudaMalloc(&d_x, xn2.size() * sizeof(bf16)));
    CK(cudaMemcpy(d_x, xn2.data(), xn2.size() * sizeof(bf16), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_fu, total_pad * I_moe * sizeof(bf16)));
    CK(cudaMemset(d_fu, 0, total_pad * I_moe * sizeof(bf16)));

    std::vector<unsigned long long> hewt(n_exp * 2);
    for (unsigned e = 0; e < n_exp; e++) {
        hewt[e*2+0] = (unsigned long long)(size_t)(d_gu + (size_t)e * 2u * I_moe * H);
        hewt[e*2+1] = (unsigned long long)(size_t)(d_dn + (size_t)e * H * I_moe);
    }
    unsigned long long* d_ewt = nullptr;
    CK(cudaMalloc(&d_ewt, hewt.size() * sizeof(unsigned long long)));
    CK(cudaMemcpy(d_ewt, hewt.data(), hewt.size()*sizeof(unsigned long long), cudaMemcpyHostToDevice));

    unsigned char* d_tab = nullptr; CK(cudaMalloc(&d_tab, tab.size()));
    CK(cudaMemcpy(d_tab, tab.data(), tab.size(), cudaMemcpyHostToDevice));

    int* d_meta = nullptr;      CK(cudaMalloc(&d_meta, (3u*n_exp + 1u) * sizeof(int)));
    unsigned* d_rt = nullptr;   CK(cudaMalloc(&d_rt, total_pad * sizeof(unsigned)));
    unsigned* d_rp = nullptr;   CK(cudaMalloc(&d_rp, total_pad * sizeof(unsigned)));
    float* d_rg = nullptr;      CK(cudaMalloc(&d_rg, total_pad * sizeof(float)));
    float* d_part = nullptr;    CK(cudaMalloc(&d_part, nslot * H * sizeof(float)));
    CK(cudaMemset(d_part, 0, nslot * H * sizeof(float)));

    const size_t arena_bytes = (size_t)PGM_ARENA_BF16 * sizeof(bf16);
    printf("dynamic smem arena: %zu bytes\n", arena_bytes);
    CK(cudaFuncSetAttribute(k_glu,  cudaFuncAttributeMaxDynamicSharedMemorySize, (int)arena_bytes));
    CK(cudaFuncSetAttribute(k_down, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)arena_bytes));

    /* op 73 first, on the WORST input a real bucket can present: `garbage` fills the
     * rows a short prompt leaves uninitialised with NaN/Inf/huge values, which is what
     * the router actually sees for rows 39..127 of a 128-row bucket. */
    if (dist == 3) {
        bf16 *d_proj = nullptr, *d_sc = nullptr, *d_pes = nullptr;
        std::vector<bf16> proj((size_t)n_exp * H), sc(H), pes(n_exp);
        for (auto& v : proj) v = __float2bfloat16(rnd() * 0.05f);
        for (auto& v : sc) v = __float2bfloat16(1.0f);
        for (auto& v : pes) v = __float2bfloat16(1.0f);
        CK(cudaMalloc(&d_proj, proj.size()*sizeof(bf16)));
        CK(cudaMemcpy(d_proj, proj.data(), proj.size()*sizeof(bf16), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&d_sc, sc.size()*sizeof(bf16)));
        CK(cudaMemcpy(d_sc, sc.data(), sc.size()*sizeof(bf16), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&d_pes, pes.size()*sizeof(bf16)));
        CK(cudaMemcpy(d_pes, pes.data(), pes.size()*sizeof(bf16), cudaMemcpyHostToDevice));
        /* real rows 0..38, then garbage: 0xFF bytes = NaN in bf16. */
        bf16* d_resid = nullptr;
        CK(cudaMalloc(&d_resid, (size_t)T * H * sizeof(bf16)));
        CK(cudaMemset(d_resid, 0xFF, (size_t)T * H * sizeof(bf16)));
        CK(cudaMemcpy(d_resid, xn2.data(), (size_t)39 * H * sizeof(bf16), cudaMemcpyHostToDevice));
        CK(cudaFuncSetAttribute(k_router, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)arena_bytes));
        k_router<<<grid, 256, arena_bytes>>>(d_tab, d_resid, d_proj, d_sc, d_pes, H, n_exp, k, T,
                                             1.0f / sqrtf((float)H), 1e-6f);
        CK(cudaDeviceSynchronize());
        printf("router(NaN-padded): ok\n");
    }

    k_align<<<1, 256>>>(d_meta, d_tab, d_rt, d_rp, d_rg, T, n_exp, k);
    CK(cudaDeviceSynchronize());
    printf("align: ok\n");

    std::vector<int> meta(3u*n_exp + 1u);
    CK(cudaMemcpy(meta.data(), d_meta, meta.size()*sizeof(int), cudaMemcpyDeviceToHost));
    int total_tiles = meta[2*n_exp + n_exp];
    int empty = 0, maxcnt = 0;
    for (unsigned e = 0; e < n_exp; e++) { int c = meta[n_exp + e]; if (!c) empty++; if (c > maxcnt) maxcnt = c; }
    const int tile_rows = MOE90_HALF ? 64 : 128; /* meta counts half-tiles under MOE90_HALF */
    printf("align: total_tiles=%d padded_rows=%d (bound %zu)  empty_experts=%d max_cnt=%d\n",
           total_tiles, total_tiles * tile_rows, total_pad, empty, maxcnt);
    if ((size_t)(total_tiles * tile_rows + 127) / 128 * 128 > total_pad) {
        printf("*** ALIGN OVERFLOWS the emitter's scratch bound ***\n");
    }

    k_glu<<<grid, 256, arena_bytes>>>(d_fu, d_x, d_ewt, d_meta, d_rt, I_moe, H, n_exp, 0u);
    CK(cudaDeviceSynchronize());
    printf("group_glu: ok\n");

    k_down<<<grid, 256, arena_bytes>>>(d_part, d_fu, d_ewt, d_meta, d_rp, d_rg, H, I_moe, n_exp);
    CK(cudaDeviceSynchronize());
    printf("group_down: ok\n");
    if (const char* dump = getenv("MOE_PART_DUMP")) { /* part is [T*k][H] by (token,slot): layout-free */
        std::vector<float> hp(nslot * H);
        CK(cudaMemcpy(hp.data(), d_part, hp.size() * sizeof(float), cudaMemcpyDeviceToHost));
        FILE* fp = fopen(dump, "wb"); fwrite(hp.data(), sizeof(float), hp.size(), fp); fclose(fp);
    }

    /* op 77: combine + sandwich norm over the scattered partials. */
    {
        bf16 *d_h1 = nullptr, *d_g = nullptr, *d_out = nullptr;
        std::vector<bf16> h1((size_t)T * H), g(H);
        for (auto& v : h1) v = __float2bfloat16(rnd() * 0.05f);
        for (auto& v : g) v = __float2bfloat16(1.0f);
        CK(cudaMalloc(&d_h1, h1.size()*sizeof(bf16)));
        CK(cudaMemcpy(d_h1, h1.data(), h1.size()*sizeof(bf16), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&d_g, g.size()*sizeof(bf16)));
        CK(cudaMemcpy(d_g, g.data(), g.size()*sizeof(bf16), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&d_out, (size_t)T * H * sizeof(bf16)));
        CK(cudaFuncSetAttribute(k_comb, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)arena_bytes));
        k_comb<<<grid, 256, arena_bytes>>>(d_out, d_part, d_h1, d_g, H, k, T, 1e-6f);
        CK(cudaDeviceSynchronize());
        printf("combine_norm: ok\n");

        /* ---- TIMING vs ROOFLINE (argv[4] = iterations; 0 = correctness only) ----
         * Served attribution (PLOW_PF_SEG_TIME) put this chain at ~89 % of prefill at
         * every rung, while the cuBLASLt dense projections — about the same FLOPs per
         * layer — take ~1/20th the time. This prices each op against its ceiling so the
         * kernel work has a number to move. FLOPs: a routed pair costs 2*(2I*H) for
         * gate|up and 2*(H*I) for down; T*k pairs per layer. */
        const int iters = argc > 4 ? atoi(argv[4]) : 0;
        if (iters > 0) {
            auto time_ms = [&](auto&& launch) {
                cudaEvent_t a, b;
                CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
                launch(); CK(cudaDeviceSynchronize());
                CK(cudaEventRecord(a));
                for (int i = 0; i < iters; i++) launch();
                CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
                float ms = 0; CK(cudaEventElapsedTime(&ms, a, b));
                return (double)ms / iters;
            };
            /* op 73 on REAL activations. It was missing from the first timing pass, and the
             * served MoE segment (~11.5 ms/layer at T=4096) is ~4.7x the sum of the other
             * four ops — the prefill router is a block-per-token loop of the decode router. */
            bf16 *rp_proj = nullptr, *rp_sc = nullptr, *rp_pes = nullptr;
            unsigned char* rp_tab = nullptr;
            {
                std::vector<bf16> proj((size_t)n_exp * H), sc(H), pes(n_exp);
                for (auto& v : proj) v = __float2bfloat16(rnd() * 0.05f);
                for (auto& v : sc) v = __float2bfloat16(1.0f);
                for (auto& v : pes) v = __float2bfloat16(1.0f);
                CK(cudaMalloc(&rp_proj, proj.size()*sizeof(bf16)));
                CK(cudaMemcpy(rp_proj, proj.data(), proj.size()*sizeof(bf16), cudaMemcpyHostToDevice));
                CK(cudaMalloc(&rp_sc, sc.size()*sizeof(bf16)));
                CK(cudaMemcpy(rp_sc, sc.data(), sc.size()*sizeof(bf16), cudaMemcpyHostToDevice));
                CK(cudaMalloc(&rp_pes, pes.size()*sizeof(bf16)));
                CK(cudaMemcpy(rp_pes, pes.data(), pes.size()*sizeof(bf16), cudaMemcpyHostToDevice));
                CK(cudaMalloc(&rp_tab, (size_t)T * k * 8));
                CK(cudaFuncSetAttribute(k_router, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)arena_bytes));
            }
            const double t_rt = time_ms([&]{ k_router<<<grid, 256, arena_bytes>>>(rp_tab, d_x, rp_proj, rp_sc, rp_pes, H, n_exp, k, T, 1.0f / sqrtf((float)H), 1e-6f); });
            if (const char* dump = getenv("MOE_RT_DUMP")) { /* bit-exactness: cmp two builds' tables */
                std::vector<unsigned char> ht((size_t)T * k * 8);
                CK(cudaMemcpy(ht.data(), rp_tab, ht.size(), cudaMemcpyDeviceToHost));
                FILE* fp = fopen(dump, "wb"); fwrite(ht.data(), 1, ht.size(), fp); fclose(fp);
            }
            const double f_rt = (double)T * 2.0 * n_exp * H; /* the [T,E,H] score GEMM */
            printf("\n  router   %8.3f ms            %6.2f TFLOP/s = %4.2f%% of peak   roof %.4f ms\n",
                   t_rt, f_rt/(t_rt*1e-3)/1e12, 100*f_rt/(t_rt*1e-3)/989e12, f_rt/989e12*1e3);
            const double t_al = time_ms([&]{ k_align<<<1, 256>>>(d_meta, d_tab, d_rt, d_rp, d_rg, T, n_exp, k); });
            const double t_gl = time_ms([&]{ k_glu<<<grid, 256, arena_bytes>>>(d_fu, d_x, d_ewt, d_meta, d_rt, I_moe, H, n_exp, 0u); });
            const double t_dn = time_ms([&]{ k_down<<<grid, 256, arena_bytes>>>(d_part, d_fu, d_ewt, d_meta, d_rp, d_rg, H, I_moe, n_exp); });
            const double t_cb = time_ms([&]{ k_comb<<<grid, 256, arena_bytes>>>(d_out, d_part, d_h1, d_g, H, k, T, 1e-6f); });
            const double pairs = (double)T * k;
            const double f_gl = pairs * 2.0 * (2.0 * I_moe) * H;   /* gate|up GEMM */
            const double f_dn = pairs * 2.0 * (double)H * I_moe;   /* down GEMM    */
            const double peak = 989e12;                            /* H100 BF16 TFLOP/s */
            const double tot = t_al + t_gl + t_dn + t_cb;
            printf("\nTIMING T=%u dist=%u (per layer, %d iters)\n", T, dist, iters);
            printf("  align    %8.3f ms  %5.1f%%   (1 block; histogram+scatter of %0.f slots)\n", t_al, 100*t_al/tot, pairs);
            printf("  glu      %8.3f ms  %5.1f%%   %6.1f TFLOP/s = %4.1f%% of peak   roof %.3f ms\n",
                   t_gl, 100*t_gl/tot, f_gl/(t_gl*1e-3)/1e12, 100*f_gl/(t_gl*1e-3)/peak, f_gl/peak*1e3);
            printf("  down     %8.3f ms  %5.1f%%   %6.1f TFLOP/s = %4.1f%% of peak   roof %.3f ms\n",
                   t_dn, 100*t_dn/tot, f_dn/(t_dn*1e-3)/1e12, 100*f_dn/(t_dn*1e-3)/peak, f_dn/peak*1e3);
            printf("  combine  %8.3f ms  %5.1f%%\n", t_cb, 100*t_cb/tot);
            printf("  TOTAL    %8.3f ms/layer  -> x30 = %.1f ms/chunk   (GEMM roof x30 = %.1f ms)\n",
                   tot, tot*30, (f_gl+f_dn)/peak*1e3*30);
        }
    }

    printf("ALL OK\n");
    return 0;
}
