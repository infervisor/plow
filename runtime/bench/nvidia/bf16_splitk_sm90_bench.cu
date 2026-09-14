/* Plow-owned Hopper BF16 static split-K experiment for Gemma-4 O/DOWN projections.
 * One CTA computes one [128,256] output tile over one contiguous K slice. The first
 * warpgroup issues TMA; two consumer warpgroups issue m64n256k16 WGMMA and write FP32
 * partials. A second kernel reduces slices in a fixed order and rounds once to BF16. */
#include <cuda.h>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublasLt.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define PLOW_NV_HOPPER 1
#define PLOW_NV_TMA_GEMM 1
#define PLOW_NV_SEG_GEMM 1
#define PLOW_NV_SEGMENTS 1
#define PLOW_NV_SEG_WS384 1
#define PGM90_UNI_BN256 1
#define PGM90_TMA_STAGES 3
#include "op_gemm.cuh"

using bf16 = __nv_bfloat16;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
    std::exit(2); } } while (0)
#define LTK(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
    std::fprintf(stderr, "cuBLASLt %s:%d: %d\n", __FILE__, __LINE__, (int)s_); \
    std::exit(2); } } while (0)

constexpr int BM = 128;
constexpr int BN = 256;
constexpr int BK = 64;
constexpr int A_STAGE_BYTES = BM * BK * (int)sizeof(bf16);
constexpr int B_STAGE_BYTES = BN * BK * (int)sizeof(bf16);
constexpr int STAGE_BYTES = A_STAGE_BYTES + B_STAGE_BYTES;
constexpr int TX_BYTES = STAGE_BYTES;
constexpr int N_FIXED = 3840;
constexpr size_t GUARD_BYTES = 4096;

template <int NS>
constexpr size_t split_smem_bytes() {
    return 2 * NS * sizeof(uint64_t) + 1024 + NS * STAGE_BYTES;
}

template <int NS, bool PRODUCER>
static __device__ void splitk_role(float* __restrict__ partial,
                                   const void* map_a, const void* map_b,
                                   int m, int n, int k, int splits, uint8_t* arena) {
    const int tid = (int)threadIdx.x;
    uint64_t* full = reinterpret_cast<uint64_t*>(arena);
    uint64_t* empty = full + NS;
    uint8_t* a_ring = static_cast<uint8_t*>(sm90_align1024(empty + NS));
    uint8_t* b_ring = a_ring + NS * A_STAGE_BYTES;

    if constexpr (PRODUCER) {
        if (tid == 0) {
            sm90_tmap_prefetch(map_a);
            sm90_tmap_prefetch(map_b);
        }
        if (tid < NS) {
            sm90_mbar_init(full + tid, 1);
            sm90_mbar_init(empty + tid, 2);
        }
    }
    __syncthreads();

    const int tiles_m = m / BM;
    int task = (int)blockIdx.x;
    const int tile_m = task % tiles_m;
    task /= tiles_m;
    const int tile_n = task % (n / BN);
    const int split = task / (n / BN);
    const int ksteps = k / BK;
    const int steps = ksteps / splits;
    const int first_step = split * steps;
    const int row0 = tile_m * BM;
    const int col0 = tile_n * BN;

    if constexpr (PRODUCER) {
        if (tid == 0) {
            for (int step = 0; step < steps; ++step) {
                const int slot = step % NS;
                if (step >= NS) sm90_mbar_wait(empty + slot, ((step / NS) + 1) & 1);
                sm90_mbar_expect(full + slot, TX_BYTES);
                const uint32_t bar = sm90_su32(full + slot);
                const int k0 = (first_step + step) * BK;
                sm90_tma2d(sm90_su32(a_ring + slot * A_STAGE_BYTES), map_a, k0, row0, bar);
                uint8_t* bs = b_ring + slot * B_STAGE_BYTES;
                sm90_tma2d(sm90_su32(bs), map_b, k0, col0, bar);
                sm90_tma2d(sm90_su32(bs + A_STAGE_BYTES), map_b, k0, col0 + 128, bar);
            }
        }
    } else {
        const int consumer = (tid >> 7) - 1;
        const int local_tid = tid & 127;
        const int warp = local_tid >> 5;
        const int lane = local_tid & 31;
        float accum[128];
        int previous = -1;
        for (int step = 0; step < steps; ++step) {
            const int slot = step % NS;
            sm90_mbar_wait(full + slot, (step / NS) & 1);
            const uint8_t* ac = a_ring + slot * A_STAGE_BYTES + consumer * 64 * 128;
            const uint8_t* bc = b_ring + slot * B_STAGE_BYTES;
            sm90_wg_fence();
#pragma unroll
            for (int sub = 0; sub < 4; ++sub) {
                wgmma_m64n256k16(accum, sm90_desc(ac + sub * 32), sm90_desc(bc + sub * 32),
                                 (step == 0 && sub == 0) ? 0 : 1);
            }
            sm90_wg_commit();
            sm90_wg_wait<1>();
            if (previous >= 0 && local_tid == 0) sm90_mbar_arrive(empty + previous);
            previous = slot;
        }
        sm90_wg_wait<0>();
        if (local_tid == 0) sm90_mbar_arrive(empty + previous);

        const int r0 = row0 + consumer * 64 + warp * 16 + (lane >> 2);
        const int c0 = col0 + 2 * (lane & 3);
        const size_t split_base = (size_t)split * m * n;
#pragma unroll
        for (int group = 0; group < BN / 8; ++group) {
#pragma unroll
            for (int hi = 0; hi < 2; ++hi) {
                const int row = r0 + 8 * hi;
                const int col = c0 + 8 * group;
                partial[split_base + (size_t)row * n + col] = accum[4 * group + 2 * hi];
                partial[split_base + (size_t)row * n + col + 1] = accum[4 * group + 2 * hi + 1];
            }
        }
    }

    __syncthreads();
    if constexpr (PRODUCER) {
        if (tid < NS) {
            sm90_mbar_inval(full + tid);
            sm90_mbar_inval(empty + tid);
        }
    }
}

template <int NS>
__global__ __maxnreg__(160) void splitk_main(float* __restrict__ partial,
                                             const void* map_a, const void* map_b,
                                             int m, int n, int k, int splits) {
    extern __shared__ uint8_t arena[];
    if (threadIdx.x < 128) {
        sm90_reg_dec(32);
        splitk_role<NS, true>(partial, map_a, map_b, m, n, k, splits, arena);
    } else {
        sm90_reg_inc(224);
        splitk_role<NS, false>(partial, map_a, map_b, m, n, k, splits, arena);
    }
}

template <int SPLITS>
__global__ void splitk_reduce(bf16* __restrict__ out, const float* __restrict__ partial,
                              size_t elements) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < elements;
         i += (size_t)gridDim.x * blockDim.x) {
        float sum = partial[i];
#pragma unroll
        for (int split = 1; split < SPLITS; ++split)
            sum = __fadd_rn(sum, partial[(size_t)split * elements + i]);
        out[i] = __float2bfloat16_rn(sum);
    }
}

__global__ __maxnreg__(160) void current_ws384(bf16* out, const void* map_a, const void* map_b,
                                                unsigned m, unsigned n, unsigned k) {
    extern __shared__ uint8_t arena[];
    if (threadIdx.x < 128) {
        sm90_reg_dec(32);
        d_gemm_sm90_tma_ws384_role<true, false>(out, map_a, map_b, nullptr, nullptr, m, n, k,
                                                0, blockIdx.x, gridDim.x,
                                                reinterpret_cast<bf16*>(arena));
    } else {
        sm90_reg_inc(224);
        d_gemm_sm90_tma_ws384_role<false, false>(out, map_a, map_b, nullptr, nullptr, m, n, k,
                                                 0, blockIdx.x, gridDim.x,
                                                 reinterpret_cast<bf16*>(arena));
    }
}

__global__ void init_values(bf16* p, size_t elements, uint32_t seed) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < elements;
         i += (size_t)gridDim.x * blockDim.x) {
        uint32_t x = (uint32_t)i ^ seed;
        x ^= x >> 16;
        x *= 0x7feb352du;
        x ^= x >> 15;
        x *= 0x846ca68bu;
        x ^= x >> 16;
        p[i] = __float2bfloat16_rn(((int)(x & 1023u) - 512) / 2048.0f);
    }
}

__global__ void evict_l2(uint32_t* p, size_t elements) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < elements;
         i += (size_t)gridDim.x * blockDim.x)
        p[i] += 1;
}

template <class T>
struct Guarded {
    uint8_t* allocation{};
    T* data{};
    size_t elements{};

    explicit Guarded(size_t count) : elements(count) {
        CK(cudaMalloc(&allocation, 2 * GUARD_BYTES + count * sizeof(T)));
        CK(cudaMemset(allocation, 0xa5, 2 * GUARD_BYTES + count * sizeof(T)));
        data = reinterpret_cast<T*>(allocation + GUARD_BYTES);
    }
    ~Guarded() { if (allocation) cudaFree(allocation); }
    Guarded(const Guarded&) = delete;
    Guarded& operator=(const Guarded&) = delete;

    void check(const char* name) const {
        std::array<uint8_t, GUARD_BYTES> guard{};
        CK(cudaMemcpy(guard.data(), allocation, GUARD_BYTES, cudaMemcpyDeviceToHost));
        for (uint8_t v : guard) if (v != 0xa5) {
            std::fprintf(stderr, "%s prefix canary overwritten\n", name); std::exit(3);
        }
        CK(cudaMemcpy(guard.data(), data + elements, GUARD_BYTES, cudaMemcpyDeviceToHost));
        for (uint8_t v : guard) if (v != 0xa5) {
            std::fprintf(stderr, "%s suffix canary overwritten\n", name); std::exit(3);
        }
    }
};

static CUtensorMap make_map(void* base, int rows, int k) {
    CUtensorMap map{};
    uint64_t dims[] = {(uint64_t)k, (uint64_t)rows};
    uint64_t strides[] = {2ull * (uint64_t)k};
    uint32_t box[] = {64, 128};
    uint32_t element_strides[] = {1, 1};
    CUresult result = cuTensorMapEncodeTiled(
        &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, base, dims, strides, box,
        element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (result != CUDA_SUCCESS) {
        std::fprintf(stderr, "cuTensorMapEncodeTiled failed: %d\n", (int)result);
        std::exit(2);
    }
    return map;
}

static void launch_reduce(bf16* out, const float* partial, size_t elements, int splits) {
    const int grid = (int)std::min<size_t>(4096, (elements + 255) / 256);
    switch (splits) {
    case 1: splitk_reduce<1><<<grid, 256>>>(out, partial, elements); break;
    case 2: splitk_reduce<2><<<grid, 256>>>(out, partial, elements); break;
    case 4: splitk_reduce<4><<<grid, 256>>>(out, partial, elements); break;
    case 8: splitk_reduce<8><<<grid, 256>>>(out, partial, elements); break;
    case 16: splitk_reduce<16><<<grid, 256>>>(out, partial, elements); break;
    default: std::fprintf(stderr, "unsupported split %d\n", splits); std::exit(2);
    }
}

static void launch_split(int stages, float* partial, bf16* out, const void* map_a,
                         const void* map_b, int m, int k, int splits) {
    const int blocks = (m / BM) * (N_FIXED / BN) * splits;
    if (stages == 3) {
        splitk_main<3><<<blocks, 384, split_smem_bytes<3>()>>>(partial, map_a, map_b, m,
                                                               N_FIXED, k, splits);
    } else {
        splitk_main<4><<<blocks, 384, split_smem_bytes<4>()>>>(partial, map_a, map_b, m,
                                                               N_FIXED, k, splits);
    }
    launch_reduce(out, partial, (size_t)m * N_FIXED, splits);
}

struct LtGemm {
    cublasLtHandle_t handle{};
    cublasLtMatmulDesc_t op{};
    cublasLtMatrixLayout_t b_layout{}, a_layout{}, c_layout{};
    cublasLtMatmulPreference_t preference{};
    cublasLtMatmulHeuristicResult_t algorithm{};
    void* workspace{};
    size_t workspace_bytes = 256u << 20;
    float alpha = 1.0f, beta = 0.0f;

    LtGemm(int m, int n, int k, const bf16* a, const bf16* b, bf16* c) {
        LTK(cublasLtCreate(&handle));
        CK(cudaMalloc(&workspace, workspace_bytes));
        LTK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        cublasOperation_t trans_b_storage = CUBLAS_OP_T, trans_a_storage = CUBLAS_OP_N;
        LTK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA,
                                           &trans_b_storage, sizeof(trans_b_storage)));
        LTK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB,
                                           &trans_a_storage, sizeof(trans_a_storage)));
        LTK(cublasLtMatrixLayoutCreate(&b_layout, CUDA_R_16BF, k, n, k));
        LTK(cublasLtMatrixLayoutCreate(&a_layout, CUDA_R_16BF, k, m, k));
        LTK(cublasLtMatrixLayoutCreate(&c_layout, CUDA_R_16BF, n, m, n));
        LTK(cublasLtMatmulPreferenceCreate(&preference));
        LTK(cublasLtMatmulPreferenceSetAttribute(preference,
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes, sizeof(workspace_bytes)));
        std::array<cublasLtMatmulHeuristicResult_t, 32> candidates{};
        int count = 0;
        LTK(cublasLtMatmulAlgoGetHeuristic(handle, op, b_layout, a_layout, c_layout, c_layout,
                                            preference, (int)candidates.size(), candidates.data(),
                                            &count));
        float best_ms = INFINITY;
        cublasLtMatmulHeuristicResult_t best_algorithm{};
        cudaEvent_t begin{}, end{};
        CK(cudaEventCreate(&begin)); CK(cudaEventCreate(&end));
        for (int i = 0; i < count; ++i) {
            if (candidates[i].state != CUBLAS_STATUS_SUCCESS) continue;
            algorithm = candidates[i];
            CK(cudaEventRecord(begin));
            run(a, b, c);
            CK(cudaEventRecord(end)); CK(cudaEventSynchronize(end));
            if (cudaPeekAtLastError() != cudaSuccess) { cudaGetLastError(); continue; }
            float ms = 0; CK(cudaEventElapsedTime(&ms, begin, end));
            if (ms < best_ms) { best_ms = ms; best_algorithm = candidates[i]; }
        }
        CK(cudaEventDestroy(begin)); CK(cudaEventDestroy(end));
        if (!std::isfinite(best_ms)) {
            std::fprintf(stderr, "no working cuBLASLt algorithm\n"); std::exit(2);
        }
        algorithm = best_algorithm;
    }

    ~LtGemm() {
        if (preference) cublasLtMatmulPreferenceDestroy(preference);
        if (c_layout) cublasLtMatrixLayoutDestroy(c_layout);
        if (a_layout) cublasLtMatrixLayoutDestroy(a_layout);
        if (b_layout) cublasLtMatrixLayoutDestroy(b_layout);
        if (op) cublasLtMatmulDescDestroy(op);
        if (workspace) cudaFree(workspace);
        if (handle) cublasLtDestroy(handle);
    }

    void run(const bf16* a, const bf16* b, bf16* c) {
        LTK(cublasLtMatmul(handle, op, &alpha, b, b_layout, a, a_layout, &beta, c,
                           c_layout, c, c_layout, &algorithm.algo, workspace, workspace_bytes, 0));
    }
};

template <class F>
static float median_ms(F&& launch, uint32_t* eviction, size_t eviction_elements) {
    for (int i = 0; i < 3; ++i) launch();
    CK(cudaDeviceSynchronize());
    std::array<float, 9> samples{};
    cudaEvent_t begin{}, end{};
    CK(cudaEventCreate(&begin)); CK(cudaEventCreate(&end));
    for (float& sample : samples) {
        evict_l2<<<512, 256>>>(eviction, eviction_elements);
        CK(cudaEventRecord(begin));
        launch();
        CK(cudaEventRecord(end)); CK(cudaEventSynchronize(end)); CK(cudaGetLastError());
        CK(cudaEventElapsedTime(&sample, begin, end));
    }
    CK(cudaEventDestroy(begin)); CK(cudaEventDestroy(end));
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

static void compare(const char* label, const Guarded<bf16>& got,
                    const std::vector<bf16>& reference) {
    std::vector<bf16> host(got.elements);
    CK(cudaMemcpy(host.data(), got.data, got.elements * sizeof(bf16), cudaMemcpyDeviceToHost));
    double error2 = 0.0, reference2 = 0.0, max_error = 0.0, max_reference = 0.0;
    for (size_t i = 0; i < host.size(); ++i) {
        const double r = __bfloat162float(reference[i]);
        const double v = __bfloat162float(host[i]);
        if (!std::isfinite(r) || !std::isfinite(v)) {
            std::fprintf(stderr, "%s produced nonfinite output\n", label); std::exit(3);
        }
        const double e = v - r;
        error2 += e * e;
        reference2 += r * r;
        max_error = std::max(max_error, std::abs(e));
        max_reference = std::max(max_reference, std::abs(r));
    }
    const double relative = std::sqrt(error2 / std::max(reference2, 1e-30));
    std::printf("correctness,%s,relL2=%.7g,max_abs=%.7g,max_ref=%.7g\n",
                label, relative, max_error, max_reference);
    if (relative > 0.006 || max_error > 0.05 + 0.02 * max_reference) std::exit(3);
}

static void run_case(int m, int k, int sms, uint32_t* eviction, size_t eviction_elements) {
    const size_t a_elements = (size_t)m * k;
    const size_t b_elements = (size_t)N_FIXED * k;
    const size_t c_elements = (size_t)m * N_FIXED;
    Guarded<bf16> a(a_elements), b(b_elements), reference(c_elements), ws(c_elements), candidate(c_elements);
    init_values<<<256, 256>>>(a.data, a_elements, 0x1234567u + (uint32_t)m);
    init_values<<<256, 256>>>(b.data, b_elements, 0x7654321u + (uint32_t)k);
    CK(cudaGetLastError());

    CUtensorMap maps[] = {make_map(a.data, m, k), make_map(b.data, N_FIXED, k)};
    CUtensorMap* device_maps{};
    CK(cudaMalloc(&device_maps, sizeof(maps)));
    CK(cudaMemcpy(device_maps, maps, sizeof(maps), cudaMemcpyHostToDevice));

    LtGemm lt(m, N_FIXED, k, a.data, b.data, reference.data);
    lt.run(a.data, b.data, reference.data);
    CK(cudaDeviceSynchronize());
    std::vector<bf16> host_reference(c_elements);
    CK(cudaMemcpy(host_reference.data(), reference.data, c_elements * sizeof(bf16),
                  cudaMemcpyDeviceToHost));

    const size_t ws_smem = PGM90_U256_ARENA * sizeof(bf16);
    CK(cudaFuncSetAttribute(current_ws384, cudaFuncAttributeMaxDynamicSharedMemorySize,
                            (int)ws_smem));
    CK(cudaFuncSetAttribute(splitk_main<3>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                            (int)split_smem_bytes<3>()));
    CK(cudaFuncSetAttribute(splitk_main<4>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                            (int)split_smem_bytes<4>()));
    auto run_ws = [&] {
        current_ws384<<<sms, 384, ws_smem>>>(ws.data, device_maps, device_maps + 1, m, N_FIXED, k);
    };
    run_ws(); CK(cudaDeviceSynchronize()); compare("WS384", ws, host_reference);

    const double flops = 2.0 * m * N_FIXED * k;
    const float lt_ms = median_ms([&] { lt.run(a.data, b.data, reference.data); }, eviction,
                                  eviction_elements);
    const float ws_ms = median_ms(run_ws, eviction, eviction_elements);
    std::printf("timing,M=%d,K=%d,backend=cuBLASLt,ms=%.6f,tflops=%.1f\n",
                m, k, lt_ms, flops / (lt_ms * 1e9));
    std::printf("timing,M=%d,K=%d,backend=WS384,ms=%.6f,tflops=%.1f,speedup_vs_lt=%.4f\n",
                m, k, ws_ms, flops / (ws_ms * 1e9), lt_ms / ws_ms);

    const std::array<int, 3> splits = m == 128 ? std::array<int, 3>{4, 8, 16}
                                      : m == 256 ? std::array<int, 3>{2, 4, 8}
                                                 : std::array<int, 3>{1, 2, 4};
    for (int split : splits) {
        Guarded<float> partial((size_t)split * c_elements);
        for (int stages : {3, 4}) {
            auto run_candidate = [&] {
                launch_split(stages, partial.data, candidate.data, device_maps, device_maps + 1,
                             m, k, split);
            };
            run_candidate(); CK(cudaDeviceSynchronize());
            char label[64];
            std::snprintf(label, sizeof(label), "splitK%d-stage%d", split, stages);
            compare(label, candidate, host_reference);
            const float ms = median_ms(run_candidate, eviction, eviction_elements);
            std::printf("timing,M=%d,K=%d,backend=splitK,split=%d,stages=%d,blocks=%d,ms=%.6f,"
                        "tflops=%.1f,speedup_vs_lt=%.4f,speedup_vs_ws=%.4f\n",
                        m, k, split, stages, (m / BM) * (N_FIXED / BN) * split, ms,
                        flops / (ms * 1e9), lt_ms / ms, ws_ms / ms);
        }
        partial.check("partial");
    }
    a.check("A"); b.check("B"); reference.check("cuBLASLt output");
    ws.check("WS384 output"); candidate.check("split-K output");
    CK(cudaFree(device_maps));
}

int main(int argc, char** argv) {
    int device = 0;
    CK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    CK(cudaGetDeviceProperties(&properties, device));
    if (properties.major != 9) {
        std::fprintf(stderr, "requires Hopper (sm_90a), found sm_%d%d\n",
                     properties.major, properties.minor);
        return 2;
    }
    const int only_m = argc > 1 ? std::atoi(argv[1]) : 0;
    const int only_k = argc > 2 ? std::atoi(argv[2]) : 0;
    if ((only_m && only_m != 128 && only_m != 256 && only_m != 512) ||
        (only_k && only_k != 8192 && only_k != 15360)) {
        std::fprintf(stderr, "usage: %s [128|256|512] [8192|15360]\n", argv[0]);
        return 2;
    }

    constexpr size_t eviction_bytes = 256u << 20;
    uint32_t* eviction{};
    CK(cudaMalloc(&eviction, eviction_bytes));
    CK(cudaMemset(eviction, 0, eviction_bytes));
    std::printf("device=%s,SMs=%d,N=%d\n", properties.name, properties.multiProcessorCount,
                N_FIXED);
    for (int m : {128, 256, 512}) {
        if (only_m && only_m != m) continue;
        for (int k : {8192, 15360}) {
            if (only_k && only_k != k) continue;
            run_case(m, k, properties.multiProcessorCount, eviction,
                     eviction_bytes / sizeof(uint32_t));
        }
    }
    CK(cudaFree(eviction));
    return 0;
}
