/* nvcc -gencode arch=compute_90a,code=sm_90a -O3 -I runtime/common -I runtime/nvidia
 *   runtime/tests/gemv_f16_sm90_test.cu -lcuda -o gemv_f16_sm90_test */
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#define PLOW_NV_SPEECH 1
#define PLOW_NV_HOPPER 1
#include "op_speech_f32.cuh"
#include "op_elementwise.cuh"

#define CK(call) do { auto e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); return 1; } } while (0)
#define DK(call) do { auto e = (call); if (e != CUDA_SUCCESS) { \
    const char* name = nullptr; cuGetErrorName(e, &name); \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, name ? name : "CUDA driver error"); return 1; } } while (0)

__global__ void project(float* c, const float* x, const unsigned short* w, const float* bias,
                        const float* res, const float* gamma, const float* beta,
                        unsigned flags, unsigned group, unsigned m, unsigned n, unsigned k) {
    d_gemv_f32out_mode(c, x, w, bias, res, gamma, beta, 1e-5f, flags, group, m, n, k,
                      blockIdx.x, gridDim.x, sp_smem, nullptr);
}

__global__ void normalize(float* out, const float* x, const float* g, const float* b,
                          unsigned rows, unsigned width, unsigned flags) {
    d_layernorm_f32(out, x, g, b, rows, width, flags, 1e-5f, blockIdx.x, gridDim.x, sp_smem);
}

#if defined(PLOW_NV_GEMM_WIDE_PROBE) && PLOW_NV_GEMM_WIDE_PROBE
template <unsigned BM, unsigned BK = 64>
__global__ void project_wide(float* c, const unsigned short* x, const unsigned short* w,
                             const float* bias, const float* res, unsigned flags, unsigned group,
                             unsigned m, unsigned n, unsigned k) {
    if (flags & 16384u)
        gemv_f16w_wide<BM, true, BK>(c, x, w, bias, group, res, flags >> 4 & 15u, m, n, k,
                               blockIdx.x, gridDim.x, sp_smem);
    else
        gemv_f16w_wide<BM, false, BK>(c, x, w, bias, group, res, flags >> 4 & 15u, m, n, k,
                                blockIdx.x, gridDim.x, sp_smem);
}
#endif

/* The M <= 8 small MMA arm called directly (what gemv_f16w_mma ran there before the streaming arm). */
__global__ void project_small(float* c, const float* x, const unsigned short* w, const float* bias,
                              const float* res, const float* gamma, const float* beta,
                              unsigned flags, unsigned m, unsigned n, unsigned k) {
    const unsigned ntiles = (n + 15u) / 16u, per = (ntiles + gridDim.x - 1) / gridDim.x, t0 = blockIdx.x * per;
    const unsigned tiles = t0 < ntiles ? min(per, ntiles - t0) : 0u;
    if (!tiles) return;
    const unsigned ks = min(max(1u, PLOW_NV_WARPS / tiles), k / 64u);
    gemv_f16w_mma_small<1>(c, x, w, (flags & 4u) ? bias : nullptr, 0u, (flags & 8u) ? res : nullptr,
                           (flags & 256u) ? gamma : nullptr, beta, 1e-5f, (flags >> 4) & 15u, m, n, k, t0, tiles,
                           ks, sp_smem);
}

/* --bench: Maya 2 depth projection shapes at M <= 8, current dispatch vs the small MMA arm, weights rotated
 * over 112 MB so every launch streams W from HBM; prints us per launch and relL2 between the arms
 * (fails above the 4e-4 small-row bound). */
static int bench() {
    CK(cudaFuncSetAttribute(project, cudaFuncAttributeMaxDynamicSharedMemorySize, 100 * 1024));
    CK(cudaFuncSetAttribute(project_small, cudaFuncAttributeMaxDynamicSharedMemorySize, 100 * 1024));
    const unsigned shapes[][3] = {{1024, 3072, 0x40f}, {3072, 1024, 0x513}, {3072, 1024, 0x503},
                                  {2048, 1024, 0x503}, {1024, 1024, 0x40b}};
    unsigned rng = 7;
    auto next = [&] { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
                     return (float)((int)(rng % 2001) - 1000) / 997.f; };
    int rc = 0;
    for (const auto& s : shapes) {
        const unsigned n = s[0], k = s[1], flags = s[2];
        const size_t wbytes = (size_t)n * k * 2;
        const unsigned copies = (unsigned)((112ull << 20) / wbytes);
        std::vector<__half> w((size_t)n * k);
        std::vector<float> x(8 * k), v(8 * n), g(k), b(k);
        for (auto& e : w) e = __float2half_rn(next() * .05f);
        for (auto& e : x) e = next();
        for (auto& e : v) e = next() * .1f;
        for (auto& e : g) e = 1.f + next() * .1f;
        for (auto& e : b) e = next() * .1f;
        unsigned short* dw; float *dx, *dv, *dg, *db, *c0, *c1;
        CK(cudaMalloc(&dw, wbytes * copies));
        for (unsigned i = 0; i < copies; i++) CK(cudaMemcpy((char*)dw + i * wbytes, w.data(), wbytes, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dx, x.size() * 4)); CK(cudaMalloc(&dv, v.size() * 4));
        CK(cudaMalloc(&dg, k * 4)); CK(cudaMalloc(&db, k * 4));
        CK(cudaMalloc(&c0, 8 * n * 4)); CK(cudaMalloc(&c1, 8 * n * 4));
        CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dv, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dg, g.data(), k * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(db, b.data(), k * 4, cudaMemcpyHostToDevice));
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        for (unsigned m : {1u, 2u, 4u, 8u}) {
            float us[2];
            for (unsigned arm = 0; arm < 2; arm++) {
                for (unsigned it = 0; it < 2 * copies + 400; it++) {
                    if (it == 2 * copies) CK(cudaEventRecord(e0));
                    const unsigned short* wi = (const unsigned short*)((char*)dw + (it % copies) * wbytes);
                    float* c = arm ? c1 : c0;
                    if (arm) project<<<132, 256, 100 * 1024>>>(c, dx, wi, dv, dv, dg, db, flags, 0, m, n, k);
                    else project_small<<<132, 256, 100 * 1024>>>(c, dx, wi, dv, dv, dg, db, flags, m, n, k);
                }
                CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
                float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); us[arm] = ms * 1000.f / 400.f;
            }
            std::vector<float> a(m * n), bb(m * n);
            CK(cudaMemcpy(a.data(), c0, m * n * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(bb.data(), c1, m * n * 4, cudaMemcpyDeviceToHost));
            double d = 0, q = 0;
            for (unsigned j = 0; j < m * n; j++) { d += ((double)a[j] - bb[j]) * ((double)a[j] - bb[j]); q += (double)a[j] * a[j]; }
            printf("BENCH M=%u N=%u K=%u flags=0x%x small_mma_us=%.2f dispatch_us=%.2f floor_us=%.2f relL2=%.3g\n", m, n, k,
                   flags, us[0], us[1], (double)wbytes / 3.1e6, sqrt(d / q));
            if (!(sqrt(d / q) <= 4e-4)) rc = 1;
        }
        CK(cudaFree(dw)); CK(cudaFree(dx)); CK(cudaFree(dv)); CK(cudaFree(dg)); CK(cudaFree(db));
        CK(cudaFree(c0)); CK(cudaFree(c1));
    }
    return rc;
}

int main(int argc, char** argv) {
    bool sanitize = false, wide = false;
    const char* node_path = nullptr;
    const char* reject_case = nullptr;
    for (int arg = 1; arg < argc; arg++) {
        if (!strcmp(argv[arg], "--bench")) return bench();
        if (!strcmp(argv[arg], "--sanitize")) sanitize = true;
        else if (!strcmp(argv[arg], "--wide")) wide = true;
        else if (!strcmp(argv[arg], "--reject") && arg + 1 < argc) reject_case = argv[++arg];
        else if (!strcmp(argv[arg], "--node") && arg + 1 < argc) { node_path = argv[++arg]; wide = true; }
        else return 2;
    }
    CUmodule node_module{};
    CUfunction node_function[2]{};
    PlowDevInst* node_instruction = nullptr;
    uint64_t* node_table = nullptr;
    if (node_path) {
        CK(cudaFree(nullptr));
        DK(cuModuleLoad(&node_module, node_path));
        DK(cuModuleGetFunction(&node_function[0], node_module, "gemm_f16_sm90_f32_node"));
        DK(cuModuleGetFunction(&node_function[1], node_module, "gemm_f16_sm90_f16_node"));
        CUdeviceptr marker;
        size_t marker_bytes;
        DK(cuModuleGetGlobal(&marker, &marker_bytes, node_module, "plow_gemm_f16_node_abi"));
        unsigned abi = 0;
        if (marker_bytes != sizeof(abi)) return 2;
        DK(cuMemcpyDtoH(&abi, marker, sizeof(abi)));
        if (abi != 1) return 2;
        for (unsigned half = 0; half < 2; half++) {
            DK(cuFuncSetAttribute(node_function[half], CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 103424));
            int occupancy = 0;
            DK(cuOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, node_function[half], 256, 103424));
            printf("NODE_OCCUPANCY half=%u blocks_per_sm=%d threads=256 dynamic_shared=103424 abi=1\n", half, occupancy);
        }
        CK(cudaMalloc(&node_instruction, sizeof(PlowDevInst)));
        CK(cudaMalloc(&node_table, 5 * sizeof(uint64_t)));
    }
    if (reject_case) {
        if (!node_path) return 2;
        PlowDevInst instruction{};
        instruction.op = PLOW_DOP_GEMV_F32_OUT;
        instruction.i[0] = 16; instruction.i[1] = 128; instruction.i[2] = 64;
        instruction.i[7] = 3 | 1024 | 8192;
        for (unsigned t = 0; t < 5; t++) instruction.t[t] = t;
        unsigned count = 5;
        if (!strcmp(reject_case, "rounded_rows")) instruction.i[0] = 0xffffffffu;
        else if (!strcmp(reject_case, "tile_count")) { instruction.i[0] = 1u<<30; instruction.i[1] = 131072; }
        else if (!strcmp(reject_case, "bias_index")) { instruction.i[0] = 1u<<26; instruction.i[3] = 1; instruction.i[7] |= 4|512; }
        else if (!strcmp(reject_case, "table_count")) count = 65536;
        else if (!strcmp(reject_case, "absent_handle")) instruction.t[0] = 0xffffu;
        else if (!strcmp(reject_case, "unsupported_flag")) instruction.i[7] |= 32;
        else return 2;
        // Deliberately unmapped disjoint spans: a missing metadata reject faults with 700,
        // distinguishable from the trap we require, or skips a wrapped zero-tile workload.
        uint64_t table[] = {0x1000000000000000ull,0x3000000000000000ull,
            0x5000000000000000ull,0x7000000000000000ull,0};
        CK(cudaMemcpy(node_instruction,&instruction,sizeof(instruction),cudaMemcpyHostToDevice));
        CK(cudaMemcpy(node_table,table,sizeof(table),cudaMemcpyHostToDevice));
        void* params[] = {&node_instruction,&node_table,&count};
        DK(cuLaunchKernel(node_function[0],2,1,1,256,1,1,103424,nullptr,params,nullptr));
        const CUresult result = cuCtxSynchronize();
        if (result != CUDA_ERROR_ILLEGAL_INSTRUCTION && result != CUDA_ERROR_LAUNCH_FAILED) {
            fprintf(stderr,"FAIL expected metadata trap for %s, got %d\n",reject_case,(int)result); return 1;
        }
        printf("REJECT_PASS case=%s cuda_code=%d\n",reject_case,(int)result);
        return 0;
    }
#if defined(PLOW_NV_GEMM_WIDE_PROBE) && PLOW_NV_GEMM_WIDE_PROBE
    if (wide && !node_path) {
        CK(cudaFuncSetAttribute(project_wide<64>, cudaFuncAttributeMaxDynamicSharedMemorySize, 100 * 1024));
        CK(cudaFuncSetAttribute(project_wide<128>, cudaFuncAttributeMaxDynamicSharedMemorySize, 100 * 1024));
        CK(cudaFuncSetAttribute(project_wide<64, 128>, cudaFuncAttributeMaxDynamicSharedMemorySize, 100 * 1024));
    }
#else
    if (wide && !node_path) return 2;
#endif
    if (!node_path) CK(cudaFuncSetAttribute(project, cudaFuncAttributeMaxDynamicSharedMemorySize, 100 * 1024));
    unsigned rng = 1701;
    auto next = [&] { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
                     return (float)((int)(rng % 2001) - 1000) / 997.f; };
    unsigned cases = 0, small = 0, small_differs = 0;
    double small_rel = 0;
    for (unsigned k : {192u, 256u, 448u, 1024u, 2048u, 3072u, 4096u}) {
        if (sanitize && k != 192u && k != 1024u && k != 2048u && k != 3072u) continue;
        /* K = 2048 at N = 1088 (= 132 x 8 + 32, a whole number of the half arm's 64-col tiles): the M <= 4
         * streaming arm's second row pass is partial. */
        const unsigned n = wide ? 128 : (k == 192 || k == 1024 || k == 3072) ? 3072 : k == 2048 ? 1088 : k == 448 ? 192 : 128,
                       max_m = 257;
        std::vector<float> x(max_m * k), bias(max_m * n), residual(max_m * n), gamma(k), beta(k);
        std::vector<__half> w(n * k);
        for (auto& v : x) v = next();
        for (auto& v : w) v = __float2half_rn(next() * .05f);
        for (auto& v : bias) v = next() * .1f;
        for (auto& v : residual) v = next() * .1f;
        for (auto& v : gamma) v = 1.f + next() * .1f;
        for (auto& v : beta) v = next() * .1f;
        float *dx, *db, *dr, *dg, *dt, *dc;
        unsigned short *dw, *dxh;
        CK(cudaMalloc(&dxh, max_m * k * 2));
        CK(cudaMalloc(&dx, x.size() * 4)); CK(cudaMalloc(&dw, w.size() * 2));
        CK(cudaMalloc(&db, bias.size() * 4)); CK(cudaMalloc(&dr, residual.size() * 4));
        CK(cudaMalloc(&dg, k * 4)); CK(cudaMalloc(&dt, k * 4)); CK(cudaMalloc(&dc, (max_m * n + 32) * 4));
        CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dw, w.data(), w.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(db, bias.data(), bias.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dr, residual.data(), residual.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dg, gamma.data(), k * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dt, beta.data(), k * 4, cudaMemcpyHostToDevice));
        if (!wide && k >= 1024u) {
            float* ln; unsigned short* lnh;
            CK(cudaMalloc(&ln, max_m * k * 4)); CK(cudaMalloc(&lnh, (max_m * k + 32) * 2));
            CK(cudaMemset(lnh, 0x7f, (max_m * k + 32) * 2));
            normalize<<<132, 256>>>(ln, dx, dg, dt, max_m, k, 0);
            normalize<<<132, 256>>>((float*)lnh, dx, dg, dt, max_m, k, 8);
            CK(cudaDeviceSynchronize());
            std::vector<float> lnf(max_m * k); std::vector<unsigned short> lnhost(max_m * k + 32);
            CK(cudaMemcpy(lnf.data(), ln, lnf.size() * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(lnhost.data(), lnh, lnhost.size() * 2, cudaMemcpyDeviceToHost));
            for (unsigned j = 0; j < lnhost.size(); j++) {
                const unsigned short expected = j < lnf.size() ? __half_as_ushort(__float2half_rn(lnf[j])) : 0x7f7f;
                if (lnhost[j] != expected) { fprintf(stderr, "FAIL LayerNorm f16 K=%u i=%u\n", k, j); return 1; }
            }
            CK(cudaFree(ln)); CK(cudaFree(lnh));
        }
        for (unsigned mode = 0; mode < (node_path ? 20u : 12u); mode++) {
            const bool optional = mode >= 12;
            if (optional && (mode & 3u) == 3u) continue;
            if (sanitize && !optional && mode != 3u && mode != 4u && mode != 6u && mode != 8u && mode != 10u) continue;
            const unsigned base_mode = optional ? (mode >= 16 ? 8 : 4) : mode;
            const bool use_bias = !optional || (mode & 1u), use_res = !optional || (mode & 2u);
            const bool ln = base_mode & 1, gelu = base_mode & 2;
            const bool half_out = base_mode & 8, half = base_mode & 12;
            if (wide && !half) continue;
            const unsigned group = gelu ? 3 : 0; /* rows 0-2 | 3-5: M = 4 crosses a bias group */
            const unsigned flags = 3 | 1024 | (use_bias ? 4 : 0) | (use_res ? 8 : 0) | (ln && !half ? 256 : 0) | (gelu ? 16 | 512 : 0)
                                   | (half ? 8192 : 0) | (half_out ? 16384 : 0);
            std::vector<double> ref(max_m * n);
            std::vector<__half> xhalf(max_m * k);
            for (unsigned row = 0; row < max_m; row++) {
                double mean = 0, var = 0;
                for (unsigned j = 0; j < k; j++) mean += x[row * k + j];
                mean /= k;
                for (unsigned j = 0; j < k; j++) { double d = x[row * k + j] - mean; var += d * d; }
                const double inv = 1 / sqrt(var / k + 1e-5);
                std::vector<float> input(k);
                for (unsigned j = 0; j < k; j++) {
                    float v = x[row * k + j];
                    if (ln) v = (float)((v - mean) * inv * gamma[j] + beta[j]);
                    input[j] = __half2float(__float2half_rn(v));
                    xhalf[row * k + j] = __float2half_rn(v);
                }
                for (unsigned col = 0; col < n; col++) {
                    double v = 0;
                    for (unsigned j = 0; j < k; j++) v += input[j] * (double)__half2float(w[col * k + j]);
                    v = (float)v;
                    if (use_bias) v = (float)(v + bias[(group ? row / group * n : 0) + col]);
                    if (use_res) v = (float)(v + residual[row * n + col]);
                    if (gelu) v = .5 * v * (1 + erf(v * .70710678118654752));
                    if (half_out) v = __half2float(__float2half_rn((float)v));
                    ref[row * n + col] = v;
                }
            }
            CK(cudaMemcpy(dxh, xhalf.data(), xhalf.size() * 2, cudaMemcpyHostToDevice));
            for (unsigned variant = 1; variant <= (wide ? 3u : 1u); variant++) {
                if (node_path && variant != 2) continue;
                if (variant == 3 && k % 128u) continue;
                std::vector<float> widest;
                for (unsigned m : {257u, 256u, 225u, 224u, 201u, 200u, 129u, 128u, 65u, 64u, 33u, 32u, 31u, 20u, 17u, 16u, 8u, 4u, 3u,
                                   2u, 1u}) {
                    if (sanitize && m != 257u && m != 224u && m != 200u && m != 33u && m != 16u && m != 4u && m != 2u) continue;
                    if (half && m < 16u) continue;
                    for (bool alias : {false, true}) {
                        if (alias && (half_out || !use_res)) continue;
                        CK(cudaMemset(dc, 0x7f, (max_m * n + 32) * 4));
                        if (alias) CK(cudaMemcpy(dc, dr, m * n * 4, cudaMemcpyDeviceToDevice));
                        if (node_path) {
                            PlowDevInst instruction{};
                            instruction.op = PLOW_DOP_GEMV_F32_OUT;
                            instruction.i[0] = m; instruction.i[1] = n; instruction.i[2] = k;
                            instruction.i[3] = group; instruction.i[7] = flags;
                            for (unsigned t = 0; t < 5; t++) instruction.t[t] = t;
                            if (!use_bias) instruction.t[3] = 0xffffu;
                            if (!use_res) instruction.t[4] = 0xffffu;
                            uint64_t table[] = {(uint64_t)dc, (uint64_t)dxh, (uint64_t)dw,
                                                use_bias ? (uint64_t)db : 0, use_res ? (uint64_t)(alias ? dc : dr) : 0};
                            CK(cudaMemcpy(node_instruction, &instruction, sizeof(instruction), cudaMemcpyHostToDevice));
                            CK(cudaMemcpy(node_table, table, sizeof(table), cudaMemcpyHostToDevice));
                            unsigned count = 5;
                            void* params[] = {&node_instruction, &node_table, &count};
                            DK(cuLaunchKernel(node_function[half_out], 2, 1, 1, 256, 1, 1, 103424, nullptr, params, nullptr));
                        } else
#if defined(PLOW_NV_GEMM_WIDE_PROBE) && PLOW_NV_GEMM_WIDE_PROBE
                        if (wide) {
                            if (variant == 1)
                                project_wide<64><<<2, 256, 100 * 1024>>>(dc, dxh, dw, db, alias ? dc : dr, flags, group, m, n, k);
                            else if (variant == 2)
                                project_wide<128><<<2, 256, 100 * 1024>>>(dc, dxh, dw, db, alias ? dc : dr, flags, group, m, n, k);
                            else
                                project_wide<64, 128><<<2, 256, 100 * 1024>>>(dc, dxh, dw, db, alias ? dc : dr, flags, group, m, n, k);
                        } else
#endif
                        project<<<132, 256, 100 * 1024>>>(dc, half ? (const float*)dxh : dx, dw, db, alias ? dc : dr, dg, dt,
                                                        flags, group, m, n, k);
                        CK(cudaDeviceSynchronize());
                        std::vector<float> got(m * n + 32);
                        bool tail_bad = false;
                        if (half_out) {
                            std::vector<unsigned short> raw(got.size());
                            CK(cudaMemcpy(raw.data(), dc, raw.size() * 2, cudaMemcpyDeviceToHost));
                            for (unsigned j = 0; j < m * n; j++) got[j] = __half2float(__ushort_as_half(raw[j]));
                            for (unsigned j = m * n; j < raw.size(); j++) tail_bad |= raw[j] != 0x7f7f;
                        } else CK(cudaMemcpy(got.data(), dc, got.size() * 4, cudaMemcpyDeviceToHost));
                        double error = 0, norm = 0;
                        for (unsigned j = 0; j < m * n; j++) {
                            error += (got[j] - ref[j]) * (got[j] - ref[j]); norm += ref[j] * ref[j];
                        }
                        const double rel = sqrt(error / norm);
                        bool bad = tail_bad || !std::isfinite(rel) || rel > 2e-4;
                        for (unsigned j = m * n; !half_out && j < got.size(); j++) {
                            unsigned bits; memcpy(&bits, &got[j], 4); bad |= bits != 0x7f7f7f7fu;
                        }
                        if (m == max_m && !alias) widest = got;
                        if (m >= 16) {
                            bad |= memcmp(got.data(), widest.data(), m * n * 4) != 0;
                        } else {
                            /* Under 16 rows the split-K MMA arm runs: another accumulation order, so the
                             * same row differs from its wider-capacity bits by rounding. Bounded, not equal. */
                            double d = 0, w2 = 0;
                            for (unsigned j = 0; j < m * n; j++) {
                                d += ((double)got[j] - widest[j]) * ((double)got[j] - widest[j]);
                                w2 += (double)widest[j] * widest[j];
                            }
                            const double across = sqrt(d / w2);
                            small++;
                            small_differs += memcmp(got.data(), widest.data(), m * n * 4) != 0;
                            small_rel = fmax(small_rel, across);
                            bad |= !std::isfinite(across) || across > 4e-4;
                        }
                        if (bad) { fprintf(stderr, "FAIL K=%u M=%u mode=%u alias=%u relL2=%g\n", k, m, mode, alias, rel); return 1; }
                        cases++;
                    }
                }
            }
        }
        CK(cudaFree(dx)); CK(cudaFree(dw)); CK(cudaFree(db)); CK(cudaFree(dr));
        CK(cudaFree(dg)); CK(cudaFree(dt)); CK(cudaFree(dc)); CK(cudaFree(dxh));
    }
    if (node_path) { CK(cudaFree(node_instruction)); CK(cudaFree(node_table)); DK(cuModuleUnload(node_module)); }
    printf("PASS %u projection cases: oracle, row tails, grouped bias, residual alias, GELU, capacity invariance from 16 rows\n", cases);
    printf("below 16 rows vs the same rows at 257: %u of %u cases differ in bits, max relL2 %.3g (bound 4e-4)\n",
           small_differs, small, small_rel);
}
