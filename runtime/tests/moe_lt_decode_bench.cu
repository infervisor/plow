/* moe_lt_decode_bench.cu — price a cuBLASLt GROUPED-GEMM MoE route at DECODE shapes on sm_90a.
 *
 * The 26B decode rungs B >= 8 already run the routed experts through the in-tree grouped wgmma
 * bodies (PLOW_GEMMA_MOE_DEC_GROUP, op_moe_sm90.cuh) inside the cooperative megakernel. A
 * cuBLASLt route (the prefill PLOW_MOE_PF_LT form: two grouped matmuls with device-side shapes
 * plus the gather / GLU / scatter glue of moe_lt_sm90.cu) can only be served by SPLITTING the
 * decode program into segments around each layer's expert pair. This harness measures, per layer
 * and at the real geometry (H 2816, I_moe 704, E 128, top-8), for a decode batch of B rows:
 *   1. the in-tree grouped bodies (align + GLU + DOWN) at that routing,
 *   2. the cuBLASLt route stage by stage (setup, gather, gate|up, GLU, down, scatter),
 *   3. relL2 of the two routes' `part` (both gate-scaled f32 [B*k][H]),
 *   4. the launch floor of a segmented decode step: a CUDA graph of 30 x (2 cooperative launches
 *      at the decode object's grid/smem + 6 glue nodes), which is what the route adds per step.
 *
 * Build: /usr/local/cuda/bin/nvcc -gencode arch=compute_90a,code=sm_90a -O3 -DPLOW_NV_GEMMA=1 \
 *        -DPLOW_NV_HOPPER=1 -DPLOW_NV_GEMV_RB=1 -I runtime/common -I runtime/nvidia \
 *        runtime/tests/moe_lt_decode_bench.cu -lcublasLt -o moeltdec
 * Run:   moeltdec <B> [dist] [iters] [grid]
 *        dist 0 = random top-8 per row, 1 = hot-skewed (32 hot experts take 75% of the slots),
 *        2 = every row the same 8 experts.
 */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublasLt.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>

#include "sm120_common.cuh"
#include "op_norm.cuh"
#include "op_elementwise.cuh"
#include "op_gemm.cuh"
#include "dev_isa.h"
#include "op_moe.cuh"
#include "moe_lt_sm90.cu"

typedef __nv_bfloat16 bf16;
#define CK(x) do { cudaError_t e_=(x); if (e_!=cudaSuccess){ \
    printf("FAIL %s at %d: %s\n",#x,__LINE__,cudaGetErrorString(e_)); exit(2);} } while(0)
#define LT(x) do { cublasStatus_t s_=(x); if (s_!=CUBLAS_STATUS_SUCCESS){ \
    printf("FAIL %s at %d: cublasLt status %d\n",#x,__LINE__,(int)s_); exit(2);} } while(0)

static uint32_t rs = 0x2468ace0u;
static float rnd() { rs ^= rs<<13; rs ^= rs>>17; rs ^= rs<<5; return (float)((int32_t)rs)/2147483648.0f; }
static unsigned rndu(unsigned n) { rs ^= rs<<13; rs ^= rs>>17; rs ^= rs<<5; return rs % n; }

extern __shared__ bf16 g_arena[];

__global__ void k_align(int* meta, const unsigned char* table, unsigned* row_token,
                        unsigned* row_partidx, float* row_gate, unsigned T, unsigned n_exp,
                        unsigned k) {
    d_moe_align_gemma_pf(meta, table, row_token, row_partidx, row_gate, T, n_exp, k, blockIdx.x);
}
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
/* Launch-floor probes: the decode object's shape (132 x 256, ~161 KiB smem) and a glue node. */
__global__ void k_coop_empty(int* p) {
    if (threadIdx.x == 0 && blockIdx.x == 0) g_arena[0] = __float2bfloat16(1.0f), p[0] = 1;
}
__global__ void k_tiny(int* p) { if (threadIdx.x == 0 && blockIdx.x == 0) p[1] = 1; }

struct Grouped {
    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatrixLayout_t w = nullptr, a = nullptr, c = nullptr;
    cublasLtMatmulAlgo_t algo{};
    size_t ws = 0;
};

static Grouped make_grouped(cublasLtHandle_t h, unsigned E, unsigned n, unsigned k,
                            const int* rows_dev, const int* n_dev, const int* k_dev,
                            unsigned average_rows, size_t ws_bytes) {
    Grouped g;
    LT(cublasLtMatmulDescCreate(&g.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const int32_t transa = CUBLAS_OP_T, pm = CUBLASLT_POINTER_MODE_DEVICE;
    LT(cublasLtMatmulDescSetAttribute(g.desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    LT(cublasLtMatmulDescSetAttribute(g.desc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pm, sizeof(pm)));
    /* Column-major views of the row-major operands (as plowrt lt.rs): W k x n, A k x m_g, C n x m_g. */
    LT(cublasLtGroupedMatrixLayoutCreate(&g.w, CUDA_R_16BF, (int)E, k_dev, n_dev, k_dev));
    LT(cublasLtGroupedMatrixLayoutCreate(&g.a, CUDA_R_16BF, (int)E, k_dev, rows_dev, k_dev));
    LT(cublasLtGroupedMatrixLayoutCreate(&g.c, CUDA_R_16BF, (int)E, n_dev, rows_dev, n_dev));
    cublasLtMatmulPreference_t pref;
    LT(cublasLtMatmulPreferenceCreate(&pref));
    LT(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes, sizeof(ws_bytes)));
    /* The 13.4 header declares these uint32_t; the library's storage is 8 bytes (lt.rs). */
    const uint64_t avg[3] = {k, n, average_rows ? average_rows : 1u};
    LT(cublasLtMatmulPreferenceSetAttribute(pref, (cublasLtMatmulPreferenceAttributes_t)13, &avg[0], 8));
    LT(cublasLtMatmulPreferenceSetAttribute(pref, (cublasLtMatmulPreferenceAttributes_t)14, &avg[1], 8));
    LT(cublasLtMatmulPreferenceSetAttribute(pref, (cublasLtMatmulPreferenceAttributes_t)15, &avg[2], 8));
    cublasLtMatmulHeuristicResult_t res[8];
    int count = 0;
    LT(cublasLtMatmulAlgoGetHeuristic(h, g.desc, g.w, g.a, g.c, g.c, pref, 8, res, &count));
    int pick = -1;
    for (int i = 0; i < count; i++)
        if (res[i].state == CUBLAS_STATUS_SUCCESS && res[i].workspaceSize <= ws_bytes) { pick = i; break; }
    if (pick < 0) { printf("FAIL: no grouped algorithm for n=%u k=%u (%d candidates)\n", n, k, count); exit(2); }
    g.algo = res[pick].algo;
    g.ws = res[pick].workspaceSize;
    printf("  grouped plan n=%u k=%u avg_rows=%u: %d candidates, picked #%d workspace %zu B waves %.2f\n",
           n, k, average_rows, count, pick, g.ws, res[pick].wavesCount);
    cublasLtMatmulPreferenceDestroy(pref);
    return g;
}

static void run_grouped(cublasLtHandle_t h, const Grouped& g, const float* scal, const void* a_ptrs,
                        const void* w_ptrs, const void* c_ptrs, void* ws, size_t ws_bytes,
                        cudaStream_t st) {
    LT(cublasLtMatmul(h, g.desc, scal, w_ptrs, g.w, a_ptrs, g.a, scal + 1, c_ptrs, g.c, (void*)c_ptrs,
                      g.c, &g.algo, ws, ws_bytes, st));
}

int main(int argc, char** argv) {
    const unsigned B     = argc > 1 ? (unsigned)atoi(argv[1]) : 16u;
    const unsigned dist  = argc > 2 ? (unsigned)atoi(argv[2]) : 0u;
    const int iters      = argc > 3 ? atoi(argv[3]) : 50;
    const unsigned grid  = argc > 4 ? (unsigned)atoi(argv[4]) : 132u;
    const unsigned H = 2816u, I = 704u, E = 128u, k = 8u;
    const size_t nslot = (size_t)B * k;
    const size_t cap = nslot + (size_t)E * 128u; /* the emitter's total_pad bound */
    printf("B=%u dist=%u iters=%d grid=%u  H=%u I=%u E=%u k=%u cap=%zu\n", B, dist, iters, grid, H, I, E, k, cap);

    /* ---- routing table [B*k] of {u32 eid, f32 gate} ---- */
    std::vector<unsigned char> tab(nslot * 8);
    std::vector<int> touched(E, 0);
    for (unsigned b = 0; b < B; b++) {
        unsigned pick[8];
        for (unsigned j = 0; j < k; j++) {
            unsigned e;
            for (;;) {
                if (dist == 2) e = 3u * j + 1u;
                else if (dist == 1) e = (rndu(4) < 3) ? rndu(32) : 32u + rndu(E - 32u);
                else e = rndu(E);
                bool dup = false;
                for (unsigned q = 0; q < j; q++) dup |= pick[q] == e;
                if (!dup) break;
            }
            pick[j] = e;
            touched[e] = 1;
            const float gate = 0.125f;
            memcpy(&tab[((size_t)b * k + j) * 8 + 0], &e, 4);
            memcpy(&tab[((size_t)b * k + j) * 8 + 4], &gate, 4);
        }
    }
    int live = 0;
    for (unsigned e = 0; e < E; e++) live += touched[e];
    const double wbytes = (double)live * ((double)2 * I * H + (double)H * I) * 2.0;
    printf("routing: %d live experts of %u -> %.1f MB of expert weights/layer, roof %.3f ms at 3.35 TB/s\n",
           live, E, wbytes / 1e6, wbytes / 3.35e12 * 1e3);

    /* ---- expert weights, distinct per expert (4 bases dealt round-robin) ---- */
    const size_t gu1 = (size_t)2u * I * H, dn1 = (size_t)H * I;
    bf16 *d_gu = nullptr, *d_dn = nullptr;
    CK(cudaMalloc(&d_gu, (size_t)E * gu1 * sizeof(bf16)));
    CK(cudaMalloc(&d_dn, (size_t)E * dn1 * sizeof(bf16)));
    {
        std::vector<bf16> bgu(4 * gu1), bdn(4 * dn1);
        for (auto& v : bgu) v = __float2bfloat16(rnd() * 0.02f);
        for (auto& v : bdn) v = __float2bfloat16(rnd() * 0.02f);
        for (unsigned e = 0; e < E; e++) {
            const unsigned b = (e * 7u + (e >> 3)) & 3u;
            CK(cudaMemcpy(d_gu + (size_t)e * gu1, bgu.data() + b * gu1, gu1 * sizeof(bf16), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_dn + (size_t)e * dn1, bdn.data() + b * dn1, dn1 * sizeof(bf16), cudaMemcpyHostToDevice));
        }
    }
    std::vector<unsigned long long> hewt(E * 2), hw_gu(E), hw_dn(E);
    for (unsigned e = 0; e < E; e++) {
        hewt[e*2+0] = hw_gu[e] = (unsigned long long)(size_t)(d_gu + (size_t)e * gu1);
        hewt[e*2+1] = hw_dn[e] = (unsigned long long)(size_t)(d_dn + (size_t)e * dn1);
    }
    unsigned long long *d_ewt = nullptr, *d_wgu = nullptr, *d_wdn = nullptr;
    CK(cudaMalloc(&d_ewt, hewt.size() * 8)); CK(cudaMemcpy(d_ewt, hewt.data(), hewt.size() * 8, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_wgu, E * 8)); CK(cudaMemcpy(d_wgu, hw_gu.data(), E * 8, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_wdn, E * 8)); CK(cudaMemcpy(d_wdn, hw_dn.data(), E * 8, cudaMemcpyHostToDevice));

    /* ---- activations and align tables ---- */
    std::vector<bf16> xn2((size_t)B * H);
    for (auto& v : xn2) v = __float2bfloat16(rnd() * 0.05f);
    bf16* d_x = nullptr; CK(cudaMalloc(&d_x, xn2.size() * sizeof(bf16)));
    CK(cudaMemcpy(d_x, xn2.data(), xn2.size() * sizeof(bf16), cudaMemcpyHostToDevice));
    unsigned char* d_tab = nullptr; CK(cudaMalloc(&d_tab, tab.size()));
    CK(cudaMemcpy(d_tab, tab.data(), tab.size(), cudaMemcpyHostToDevice));
    int* d_meta = nullptr;    CK(cudaMalloc(&d_meta, (3u * E + 2u) * sizeof(int)));
    unsigned* d_rt = nullptr; CK(cudaMalloc(&d_rt, cap * 4));
    unsigned* d_rp = nullptr; CK(cudaMalloc(&d_rp, cap * 4));
    float* d_rg = nullptr;    CK(cudaMalloc(&d_rg, cap * 4));
    bf16* d_fu = nullptr;     CK(cudaMalloc(&d_fu, cap * I * sizeof(bf16)));
    float *d_part_a = nullptr, *d_part_b = nullptr;
    CK(cudaMalloc(&d_part_a, nslot * H * 4)); CK(cudaMemset(d_part_a, 0, nslot * H * 4));
    CK(cudaMalloc(&d_part_b, nslot * H * 4)); CK(cudaMemset(d_part_b, 0, nslot * H * 4));

    const size_t arena_bytes = (size_t)PGM_ARENA_BF16 * sizeof(bf16);
    CK(cudaFuncSetAttribute(k_glu,  cudaFuncAttributeMaxDynamicSharedMemorySize, (int)arena_bytes));
    CK(cudaFuncSetAttribute(k_down, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)arena_bytes));
    CK(cudaFuncSetAttribute(k_coop_empty, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)arena_bytes));
    printf("in-tree arena %zu B (MOE90_HALF=%d)\n", arena_bytes, (int)MOE90_HALF);

    k_align<<<1, 256>>>(d_meta, d_tab, d_rt, d_rp, d_rg, B, E, k);
    CK(cudaDeviceSynchronize());
    std::vector<int> meta(3u * E + 2u);
    CK(cudaMemcpy(meta.data(), d_meta, meta.size() * 4, cudaMemcpyDeviceToHost));
    const int total_tiles = meta[3 * E];
    const int tile_rows = MOE90_HALF ? 64 : 128;
    const int extent = meta[E - 1] + meta[2 * E - 1];
    printf("align: %d half-tiles, gathered extent %d rows (bound %zu), GLU work items %d, DOWN items %d on %u blocks\n",
           total_tiles, extent, cap, (total_tiles + 1) / 2 * (int)((I + 127) / 128),
           (total_tiles + 1) / 2 * (int)((H + 255) / 256), grid);
    (void)tile_rows;

    /* ---- in-tree grouped route, correctness pass ---- */
    k_glu<<<grid, 256, arena_bytes>>>(d_fu, d_x, d_ewt, d_meta, d_rt, I, H, E, 0u);
    k_down<<<grid, 256, arena_bytes>>>(d_part_a, d_fu, d_ewt, d_meta, d_rp, d_rg, H, I, E);
    CK(cudaDeviceSynchronize());

    /* ---- cuBLASLt route ---- */
    cublasLtHandle_t lt; LT(cublasLtCreate(&lt));
    const size_t ws_bytes = 64u << 20;
    void* d_ws = nullptr; CK(cudaMalloc(&d_ws, ws_bytes));
    float* d_scal = nullptr; { const float ab[2] = {1.0f, 0.0f}; CK(cudaMalloc(&d_scal, 8)); CK(cudaMemcpy(d_scal, ab, 8, cudaMemcpyHostToDevice)); }
    /* tables: i32 [rows | 2I | H | I] x E, then u64 [xs | gu | fu | dn] x E (moe_lt.rs) */
    int* d_tabi = nullptr; CK(cudaMalloc(&d_tabi, E * 16 + E * 32));
    {
        std::vector<int> c(4 * E);
        for (unsigned e = 0; e < E; e++) { c[e] = 0; c[E + e] = (int)(2 * I); c[2 * E + e] = (int)H; c[3 * E + e] = (int)I; }
        CK(cudaMemcpy(d_tabi, c.data(), c.size() * 4, cudaMemcpyHostToDevice));
    }
    unsigned long long* d_ptrs = (unsigned long long*)(d_tabi + 4 * E);
    bf16* d_xs = nullptr; CK(cudaMalloc(&d_xs, cap * H * sizeof(bf16)));
    bf16* d_gubuf = nullptr; CK(cudaMalloc(&d_gubuf, cap * 2 * I * sizeof(bf16)));
    bf16* d_fu2 = nullptr; CK(cudaMalloc(&d_fu2, cap * I * sizeof(bf16)));
    const unsigned avg_rows = (unsigned)((nslot + E - 1) / E);
    Grouped g_gu = make_grouped(lt, E, 2 * I, H, d_tabi, d_tabi + E, d_tabi + 2 * E, avg_rows, ws_bytes);
    Grouped g_dn = make_grouped(lt, E, H, I, d_tabi, d_tabi + 2 * E, d_tabi + 3 * E, avg_rows, ws_bytes);
    int nblk_glue = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nblk_glue, plow_moe_lt_gather, 256, 0));
    { int dev = 0, sms = 0; CK(cudaGetDevice(&dev)); CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev)); nblk_glue = (nblk_glue < 1 ? 1 : nblk_glue) * sms; }
    printf("glue grid %d blocks\n", nblk_glue);

    cudaStream_t st; CK(cudaStreamCreate(&st));
    auto lt_setup   = [&]{ plow_moe_lt_setup<<<1, 256, 0, st>>>(d_tabi, d_ptrs, d_meta, (unsigned long long)(size_t)d_xs, (unsigned long long)(size_t)d_gubuf, (unsigned long long)(size_t)d_fu2, (unsigned long long)(size_t)d_xs, E, H, I); };
    auto lt_gather  = [&]{ plow_moe_lt_gather<<<nblk_glue, 256, 0, st>>>(d_xs, d_x, d_rt, d_meta, E, H, (unsigned)nblk_glue); };
    auto lt_gemm1   = [&]{ run_grouped(lt, g_gu, d_scal, d_ptrs, d_wgu, d_ptrs + E, d_ws, ws_bytes, st); };
    auto lt_glu     = [&]{ plow_moe_lt_glu<<<nblk_glue, 256, 0, st>>>(d_fu2, d_gubuf, d_meta, E, I, 0u, (unsigned)nblk_glue); };
    auto lt_gemm2   = [&]{ run_grouped(lt, g_dn, d_scal, d_ptrs + 2 * E, d_wdn, d_ptrs + 3 * E, d_ws, ws_bytes, st); };
    auto lt_scatter = [&]{ plow_moe_lt_scatter<<<nblk_glue, 256, 0, st>>>(d_part_b, d_xs, d_rp, d_rg, d_meta, E, H, (unsigned)nblk_glue); };
    auto lt_all = [&]{ lt_setup(); lt_gather(); lt_gemm1(); lt_glu(); lt_gemm2(); lt_scatter(); };
    lt_all();
    CK(cudaStreamSynchronize(st));
    CK(cudaGetLastError());

    /* numerics: both routes' part over the live slots */
    {
        std::vector<float> pa(nslot * H), pb(nslot * H);
        CK(cudaMemcpy(pa.data(), d_part_a, pa.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(pb.data(), d_part_b, pb.size() * 4, cudaMemcpyDeviceToHost));
        double num = 0, den = 0, maxabs = 0;
        for (size_t i = 0; i < pa.size(); i++) {
            const double d = (double)pa[i] - pb[i];
            num += d * d; den += (double)pa[i] * pa[i];
            if (fabs(d) > maxabs) maxabs = fabs(d);
        }
        printf("numerics: part relL2 (Lt vs in-tree) = %.3e, max|d| = %.3e, |part| rms %.3e\n",
               sqrt(num / (den > 0 ? den : 1)), maxabs, sqrt(den / pa.size()));
    }

    auto time_ms = [&](auto&& launch) {
        cudaEvent_t a, b;
        CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        launch(); CK(cudaStreamSynchronize(st));
        CK(cudaEventRecord(a, st));
        for (int i = 0; i < iters; i++) launch();
        CK(cudaEventRecord(b, st)); CK(cudaEventSynchronize(b));
        float ms = 0; CK(cudaEventElapsedTime(&ms, a, b));
        return (double)ms / iters;
    };
    const double t_al = time_ms([&]{ k_align<<<1, 256, 0, st>>>(d_meta, d_tab, d_rt, d_rp, d_rg, B, E, k); });
    const double t_gl = time_ms([&]{ k_glu<<<grid, 256, arena_bytes, st>>>(d_fu, d_x, d_ewt, d_meta, d_rt, I, H, E, 0u); });
    const double t_dn = time_ms([&]{ k_down<<<grid, 256, arena_bytes, st>>>(d_part_a, d_fu, d_ewt, d_meta, d_rp, d_rg, H, I, E); });
    const double t_s = time_ms(lt_setup), t_g = time_ms(lt_gather), t_m1 = time_ms(lt_gemm1),
                 t_u = time_ms(lt_glu), t_m2 = time_ms(lt_gemm2), t_c = time_ms(lt_scatter),
                 t_lt = time_ms(lt_all);
    const double gu_b = (double)live * 2 * I * H * 2, dn_b = (double)live * H * I * 2;
    printf("\nPER LAYER, ms (B=%u, %d live experts, %d iters)\n", B, live, iters);
    printf("  in-tree  align %.4f  GLU %.4f (%.0f GB/s of weight)  DOWN %.4f (%.0f GB/s)  = %.4f\n",
           t_al, t_gl, gu_b / (t_gl * 1e-3) / 1e9, t_dn, dn_b / (t_dn * 1e-3) / 1e9, t_al + t_gl + t_dn);
    printf("  cuBLASLt setup %.4f gather %.4f gate|up %.4f (%.0f GB/s) glu %.4f down %.4f (%.0f GB/s) scatter %.4f  sum %.4f  chained %.4f\n",
           t_s, t_g, t_m1, gu_b / (t_m1 * 1e-3) / 1e9, t_u, t_m2, dn_b / (t_m2 * 1e-3) / 1e9, t_c,
           t_s + t_g + t_m1 + t_u + t_m2 + t_c, t_lt);
    printf("  x30 layers: in-tree %.3f ms/step, cuBLASLt chained %.3f ms/step (GEMMs only %.3f)\n",
           30 * (t_al + t_gl + t_dn), 30 * t_lt, 30 * (t_m1 + t_m2));

    /* ---- launch floor of a segmented decode step ---- */
    {
        int* d_flag = nullptr; CK(cudaMalloc(&d_flag, 64));
        auto coop = [&](cudaStream_t s) {
            void* args[] = {&d_flag};
            CK(cudaLaunchCooperativeKernel((const void*)k_coop_empty, dim3(grid), dim3(256), args, arena_bytes, s));
        };
        auto capture = [&](int layers, int glue_per_layer, int coop_per_layer) {
            cudaGraph_t graph; cudaGraphExec_t exec;
            cudaStream_t cs; CK(cudaStreamCreate(&cs));
            CK(cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal));
            for (int l = 0; l < layers; l++) {
                for (int c = 0; c < coop_per_layer; c++) {
                    coop(cs);
                    if (c == 0) for (int j = 0; j < glue_per_layer; j++) k_tiny<<<nblk_glue, 256, 0, cs>>>(d_flag);
                }
            }
            CK(cudaStreamEndCapture(cs, &graph));
            CK(cudaGraphInstantiate(&exec, graph, 0));
            const double ms = time_ms([&]{ CK(cudaGraphLaunch(exec, st)); });
            cudaGraphExecDestroy(exec); cudaGraphDestroy(graph); cudaStreamDestroy(cs);
            return ms;
        };
        const double one = capture(1, 0, 1);
        const double seg = capture(30, 0, 2);
        const double full = capture(30, 6, 2);
        printf("\nLAUNCH FLOOR (graph): 1 cooperative launch %.4f ms; 30 x 2 cooperative %.3f ms; 30 x (2 cooperative + 6 glue nodes) %.3f ms per step\n",
               one, seg, full);
    }
    printf("ALL OK\n");
    return 0;
}
