// Gemma-4 gate/up projection experiment. --fp8 / --fp8-wide compare promoted
// FP8 64x64 / 64x128 tiles against the production promoted 128x128 tile.
// Exact BF16 gate/up projection experiment. The control is two shipped WS384 BF16 GEMMs
// followed by d_glu; the candidate shares A staging and fuses the same BF16-boundary GeGLU.
//
// Build NS=3/4 independently:
//   nvcc -std=c++17 -gencode arch=compute_90a,code=sm_90a -O3 \
//     -I runtime/common -I runtime/nvidia -include cstdint -Xptxas=-v \
//     -DPLOW_GLU_NS=3 runtime/nvidia/experiments/gemma4_ws384_fused_glu.cu \
//     -lcuda -lcublasLt -o /tmp/gemma4_ws384_glu_ns3
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cublasLt.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#ifndef PLOW_GLU_NS
#define PLOW_GLU_NS 3
#endif
#if PLOW_GLU_NS != 3 && PLOW_GLU_NS != 4
#error "PLOW_GLU_NS must be 3 or 4"
#endif

#define PLOW_NV_HOPPER 1
#define PLOW_NV_THREADS 256u
#define PLOW_NV_TMA_GEMM 1
#define PLOW_NV_W8A8 1
#define PGM90_UNI_BN256 1
#define PLOW_NV_SEG_GEMM 1
#define PLOW_NV_SEGMENTS 1
#define PLOW_NV_SEG_WS384 1
#define PGM90_WS384_BN 256
#define PGM90_UNI256_NS 3
#define PGM90_WS384_SMEPI 0
#define PGM90_WS384_PREFETCH 1
#define PGM90_WS384_ISSUE_CURSOR 1
#define PLOW_NV_GEMMA 1
#define PLOW_NV_GEMMA4_GLU_FP8_PROMOTE 1
#define PLOW_NV_GEMMA4_GLU_FP8_PROMOTE_STAGES 1

#include "sm120_common.cuh"
#include "op_gemm.cuh"
#include "op_elementwise.cuh"

using bf16 = __nv_bfloat16;

#define CK(x)                                                                                   \
    do {                                                                                        \
        cudaError_t e_ = (x);                                                                   \
        if (e_ != cudaSuccess) {                                                                \
            std::fprintf(stderr, "%s: %s at line %d\n", #x, cudaGetErrorString(e_), __LINE__); \
            std::exit(2);                                                                       \
        }                                                                                       \
    } while (0)
#define CKD(x)                                                                                  \
    do {                                                                                        \
        CUresult e_ = (x);                                                                      \
        if (e_ != CUDA_SUCCESS) {                                                               \
            const char* s_ = nullptr;                                                           \
            cuGetErrorString(e_, &s_);                                                          \
            std::fprintf(stderr, "%s: %s at line %d\n", #x, s_ ? s_ : "driver error",        \
                         __LINE__);                                                              \
            std::exit(2);                                                                       \
        }                                                                                       \
    } while (0)

constexpr int GLU_BM = 128;
constexpr int GLU_BN = 128;
constexpr int GLU_BK = 64;
constexpr int GLU_TILE_BYTES = GLU_BM * GLU_BK * sizeof(bf16);
constexpr int GLU_TX_BYTES = 3 * GLU_TILE_BYTES;
constexpr int GLU_MBAR_BYTES = 2 * PLOW_GLU_NS * sizeof(uint64_t);
constexpr int GLU_SMEM = GLU_MBAR_BYTES + PLOW_GLU_NS * GLU_TX_BYTES + 1024;

template <bool Producer>
__device__ void fused_glu_body(bf16* __restrict__ out, const void* map_a, const void* map_g,
                               const void* map_u, unsigned m, unsigned n, unsigned k,
                               unsigned slice, unsigned nblk, void* raw_arena) {
    uint64_t* full = static_cast<uint64_t*>(raw_arena);
    uint64_t* empty = full + PLOW_GLU_NS;
    uint8_t* base = static_cast<uint8_t*>(sm90_align1024(empty + PLOW_GLU_NS));
    uint8_t* as = base;
    uint8_t* gs = as + PLOW_GLU_NS * GLU_TILE_BYTES;
    uint8_t* us = gs + PLOW_GLU_NS * GLU_TILE_BYTES;
    const int tid = threadIdx.x;

    if constexpr (Producer) {
        if (tid == 0) {
            sm90_tmap_prefetch(map_a);
            sm90_tmap_prefetch(map_g);
            sm90_tmap_prefetch(map_u);
        }
        if (tid < PLOW_GLU_NS) {
            sm90_mbar_init(full + tid, 1);
            sm90_mbar_init(empty + tid, 2);
        }
    }
    __syncthreads();

    const int tiles_m = (m + GLU_BM - 1) / GLU_BM;
    const int tiles_n = (n + GLU_BN - 1) / GLU_BN;
    const int ntiles = tiles_m * tiles_n;
    const int ksteps = (k + GLU_BK - 1) / GLU_BK;

    if constexpr (Producer) {
        if (tid == 0) {
            int issued = 0;
            for (int tile = slice; tile < ntiles; tile += nblk) {
                int tmi, tni;
                sm90_tile_remap(tile, tiles_m, tiles_n, &tmi, &tni);
                const int tm = tmi * GLU_BM;
                const int tn = tni * GLU_BN;
                for (int ks = 0; ks < ksteps; ++ks, ++issued) {
                    const int stage = issued % PLOW_GLU_NS;
                    if (issued >= PLOW_GLU_NS)
                        sm90_mbar_wait(empty + stage, ((issued / PLOW_GLU_NS) + 1) & 1);
                    sm90_mbar_expect(full + stage, GLU_TX_BYTES);
                    const uint32_t bar = sm90_su32(full + stage);
                    sm90_tma2d(sm90_su32(as + stage * GLU_TILE_BYTES), map_a, ks * GLU_BK, tm,
                               bar);
                    sm90_tma2d(sm90_su32(gs + stage * GLU_TILE_BYTES), map_g, ks * GLU_BK, tn,
                               bar);
                    sm90_tma2d(sm90_su32(us + stage * GLU_TILE_BYTES), map_u, ks * GLU_BK, tn,
                               bar);
                }
            }
        }
    } else {
        const int consumer = (tid >> 7) - 1;
        const int lt = tid & 127;
        const int warp = lt >> 5;
        const int lane = lt & 31;
        int step = 0;
        for (int tile = slice; tile < ntiles; tile += nblk) {
            int tmi, tni;
            sm90_tile_remap(tile, tiles_m, tiles_n, &tmi, &tni);
            const int tm = tmi * GLU_BM;
            const int tn = tni * GLU_BN;
            float gate[GLU_BN / 2];
            float up[GLU_BN / 2];
            int previous = -1;
            for (int ks = 0; ks < ksteps; ++ks, ++step) {
                const int stage = step % PLOW_GLU_NS;
                sm90_mbar_wait(full + stage, (step / PLOW_GLU_NS) & 1);
                const uint8_t* a = as + stage * GLU_TILE_BYTES + consumer * 64 * 128;
                const uint8_t* g = gs + stage * GLU_TILE_BYTES;
                const uint8_t* u = us + stage * GLU_TILE_BYTES;
                sm90_wg_fence();
#pragma unroll
                for (int sub = 0; sub < 4; ++sub) {
                    const int accumulate = (ks == 0 && sub == 0) ? 0 : 1;
                    wgmma_m64n128k16(gate, sm90_desc(a + sub * 32), sm90_desc(g + sub * 32),
                                     accumulate);
                    wgmma_m64n128k16(up, sm90_desc(a + sub * 32), sm90_desc(u + sub * 32),
                                     accumulate);
                }
                sm90_wg_commit();
                sm90_wg_wait<1>();
                if (previous >= 0 && lt == 0) sm90_mbar_arrive(empty + previous);
                previous = stage;
            }
            sm90_wg_wait<0>();
            if (previous >= 0 && lt == 0) sm90_mbar_arrive(empty + previous);

            const int r0 = tm + consumer * 64 + warp * 16 + (lane >> 2);
            const int c0 = tn + 2 * (lane & 3);
#pragma unroll
            for (int group = 0; group < GLU_BN / 8; ++group) {
#pragma unroll
                for (int hi = 0; hi < 2; ++hi) {
                    const int row = r0 + 8 * hi;
                    const int col = c0 + 8 * group;
                    if (row >= static_cast<int>(m) || col + 1 >= static_cast<int>(n)) continue;
#pragma unroll
                    for (int pair = 0; pair < 2; ++pair) {
                        const int ai = 4 * group + 2 * hi + pair;
                        const float rounded_gate =
                            __bfloat162float(__float2bfloat16(gate[ai]));
                        const float rounded_up = __bfloat162float(__float2bfloat16(up[ai]));
                        float activated = act_gelu_tanh(rounded_gate);
#if defined(PLOW_NV_GEMMA_GLU_BF16) && PLOW_NV_GEMMA_GLU_BF16
                        activated = __bfloat162float(__float2bfloat16(activated));
#endif
                        out[static_cast<size_t>(row) * n + col + pair] =
                            __float2bfloat16(activated * rounded_up);
                    }
                }
            }
        }
    }
    __syncthreads();
    if constexpr (Producer) {
        if (tid < PLOW_GLU_NS) {
            sm90_mbar_inval(full + tid);
            sm90_mbar_inval(empty + tid);
        }
    }
    __syncthreads();
}

__global__ __maxnreg__(160) void fused_glu(bf16* out, const void* map_a, const void* map_g,
                                            const void* map_u, unsigned m, unsigned n,
                                            unsigned k) {
    extern __shared__ uint8_t arena[];
    if (threadIdx.x < 128) {
        sm90_reg_dec(32);
        fused_glu_body<true>(out, map_a, map_g, map_u, m, n, k, blockIdx.x, gridDim.x, arena);
    } else {
        sm90_reg_inc(224);
        fused_glu_body<false>(out, map_a, map_g, map_u, m, n, k, blockIdx.x, gridDim.x, arena);
    }
}

__global__ __maxnreg__(160) void shipped_ws384(bf16* out, const void* map_a, const void* map_b,
                                               unsigned m, unsigned n, unsigned k) {
    extern __shared__ uint8_t arena[];
    if (threadIdx.x < 128) {
        sm90_reg_dec(32);
        d_gemm_sm90_tma_ws384_role<true, false>(out, map_a, map_b, nullptr, nullptr, m, n, k, 0,
                                                blockIdx.x, gridDim.x,
                                                reinterpret_cast<bf16*>(arena));
    } else {
        sm90_reg_inc(224);
        d_gemm_sm90_tma_ws384_role<false, false>(out, map_a, map_b, nullptr, nullptr, m, n, k,
                                                 0, blockIdx.x, gridDim.x,
                                                 reinterpret_cast<bf16*>(arena));
    }
}

__global__ void shipped_glu(bf16* out, const bf16* gate, const bf16* up, unsigned count) {
    d_glu(out, gate, up, count, PLOW_ACT_GELU_TANH_, blockIdx.x, gridDim.x);
}

__global__ void combined_glu(bf16* out, const bf16* projections, unsigned rows) {
    constexpr unsigned width = 15360;
    const unsigned count = rows * width;
    const unsigned stride = gridDim.x * blockDim.x * 8;
    auto offset = [](unsigned i) { return size_t(i / width) * 2 * width + i % width; };
    auto glu8 = [](const bf16v8& gate, const bf16v8& up) {
        bf16v8 result;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            float activated = act_gelu_tanh_pf(__bfloat162float(gate.x[j]));
#if defined(PLOW_NV_GEMMA_GLU_BF16) && PLOW_NV_GEMMA_GLU_BF16
            activated = __bfloat162float(__float2bfloat16(activated));
#endif
            result.x[j] = __float2bfloat16(activated * __bfloat162float(up.x[j]));
        }
        return result;
    };
    unsigned i = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    for (; i + 3 * stride < count; i += 4 * stride) {
        bf16v8 gate[4], up[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const size_t source = offset(i + j * stride);
            gate[j] = ld_glob8(projections + source);
            up[j] = ld_glob8(projections + source + width);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j)
            st_glob8(out + i + j * stride, glu8(gate[j], up[j]));
    }
    for (; i < count; i += stride) {
        const size_t source = offset(i);
        st_glob8(out + i, glu8(ld_glob8(projections + source),
                               ld_glob8(projections + source + width)));
    }
}

__global__ void quant_plain(bf16* output, uint8_t* quantized, float* scales, unsigned rows) {
    __shared__ float part[8];
    d_quant_fp8(quantized, output, scales, rows, 15360, blockIdx.x, gridDim.x,
                 nullptr, nullptr, 0, part);
}

template <unsigned WPR>
__global__ void quant_glu(bf16* output, uint8_t* quantized, float* scales,
                          const bf16* gate, const bf16* up, unsigned rows) {
    __shared__ float part[8];
    if constexpr (WPR == 0)
        d_quant_fp8(quantized, output, scales, rows, 15360, blockIdx.x, gridDim.x,
                     gate, up, PLOW_ACT_GELU_TANH_, part);
    else
        d_quant_fp8_wpr<WPR, true>(quantized, output, scales, rows, 15360, blockIdx.x,
                                   gridDim.x, part, gate, up, PLOW_ACT_GELU_TANH_);
}

__global__ void quant_glu_cached(bf16* output, uint8_t* quantized, float* scales,
                                  const bf16* gate, const bf16* up, unsigned rows) {
    __shared__ float part[8];
    d_glu_quant_fp8_cached(output, quantized, scales, gate, up, rows,
                            blockIdx.x, gridDim.x, part);
}

static bool compare_glu_quant(bf16* output, const bf16* gate, const bf16* up, unsigned rows, int grid) {
    const size_t count = size_t(rows) * 15360;
    uint8_t* quantized;
    float* scales;
    CK(cudaMalloc(&quantized, count));
    CK(cudaMalloc(&scales, size_t(rows) * sizeof(float)));
    auto launch = [&](int arm) {
        if (arm == 0) quant_glu<0><<<grid,256>>>(output, quantized, scales, gate, up, rows);
        else if (arm == 1) quant_glu<4><<<grid,256>>>(output, quantized, scales, gate, up, rows);
        else if (arm == 2) quant_glu<8><<<grid,256>>>(output, quantized, scales, gate, up, rows);
        else quant_glu_cached<<<grid,256>>>(output, quantized, scales, gate, up, rows);
        CK(cudaGetLastError());
    };
    std::vector<bf16> reference(count), actual(count);
    std::vector<uint8_t> rq(count), aq(count);
    std::vector<float> rs(rows), as(rows);
    bool correct = true;
    for (int arm = -1; arm < 4; ++arm) {
        if (arm == -1) {
            shipped_glu<<<grid,256>>>(output, gate, up, count);
            quant_plain<<<grid,256>>>(output, quantized, scales, rows);
            CK(cudaGetLastError());
        } else launch(arm);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(actual.data(), output, count * sizeof(bf16), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(aq.data(), quantized, count, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(as.data(), scales, size_t(rows) * sizeof(float), cudaMemcpyDeviceToHost));
        if (arm == -1) { reference = actual; rq = aq; rs = as; }
        const bool bf16_equal = std::memcmp(reference.data(), actual.data(), count * sizeof(bf16)) == 0;
        const bool fp8_equal = rq == aq;
        const bool scales_equal = std::memcmp(rs.data(), as.data(), size_t(rows) * sizeof(float)) == 0;
        bool finite = true;
        for (size_t i = 0; i < count; ++i) finite &= std::isfinite(__bfloat162float(actual[i]));
        for (float scale : as) finite &= std::isfinite(scale) && scale > 0;
        correct &= bf16_equal && fp8_equal && scales_equal && finite;
        std::printf("glu_quant_arm=%d bf16_equal=%d fp8_equal=%d scales_equal=%d finite=%d\n",
                    arm, bf16_equal, fp8_equal, scales_equal, finite);
    }
    cudaEvent_t begin, end;
    CK(cudaEventCreate(&begin));
    CK(cudaEventCreate(&end));
    std::vector<float> times[4];
    const int orders[8][4] = {{0,1,2,3},{1,2,3,0},{2,3,0,1},{3,0,1,2},
                              {3,2,1,0},{0,3,2,1},{1,0,3,2},{2,1,0,3}};
    for (int round = 0; round < 32; ++round) {
        for (int arm : orders[round % 8]) {
            CK(cudaEventRecord(begin));
            launch(arm);
            CK(cudaEventRecord(end));
            CK(cudaEventSynchronize(end));
            float ms;
            CK(cudaEventElapsedTime(&ms, begin, end));
            if (round >= 8) {
                times[arm].push_back(ms);
                std::printf("glu_quant_round=%d arm=%d ms=%.6f\n", round - 8, arm, ms);
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
    for (int arm = 0; arm < 4; ++arm) {
        std::sort(times[arm].begin(), times[arm].end());
        std::printf("glu_quant_arm=%d p50_ms=%.6f\n", arm, (times[arm][11] + times[arm][12]) / 2);
    }
    CK(cudaEventDestroy(begin));
    CK(cudaEventDestroy(end));
    cudaFree(quantized);
    cudaFree(scales);
    return correct;
}

static CUtensorMap make_map(void* base, unsigned rows, unsigned k) {
    CUtensorMap map{};
    uint64_t dims[] = {k, rows};
    uint64_t strides[] = {uint64_t(k) * sizeof(bf16)};
    uint32_t box[] = {GLU_BK, GLU_BM};
    uint32_t steps[] = {1, 1};
    CKD(cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, base, dims, strides,
                               box, steps, CU_TENSOR_MAP_INTERLEAVE_NONE,
                               CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return map;
}

template <typename T>
static T* upload(const std::vector<T>& host) {
    T* device;
    CK(cudaMalloc(&device, host.size() * sizeof(T)));
    CK(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    return device;
}

static float timed(bool candidate, int iterations, int grid, int smem, bf16* out, bf16* gate,
                   bf16* up, const CUtensorMap* maps, unsigned m, unsigned n, unsigned k) {
    cudaEvent_t begin, end;
    CK(cudaEventCreate(&begin));
    CK(cudaEventCreate(&end));
    CK(cudaEventRecord(begin));
    for (int i = 0; i < iterations; ++i) {
        if (candidate) {
            fused_glu<<<grid, 384, smem>>>(out, maps, maps + 1, maps + 2, m, n, k);
        } else {
            shipped_ws384<<<grid, 384, 2 * PGM90_WS384_ARENA>>>(gate, maps, maps + 1, m, n, k);
            shipped_ws384<<<grid, 384, 2 * PGM90_WS384_ARENA>>>(up, maps, maps + 2, m, n, k);
            shipped_glu<<<grid, 256>>>(out, gate, up, size_t(m) * n);
        }
    }
    CK(cudaEventRecord(end));
    CK(cudaEventSynchronize(end));
    float elapsed;
    CK(cudaEventElapsedTime(&elapsed, begin, end));
    CK(cudaEventDestroy(begin));
    CK(cudaEventDestroy(end));
    return elapsed / iterations;
}

static void print_resources(const char* name, const void* kernel, int threads, int smem) {
    cudaFuncAttributes attr{};
    CK(cudaFuncGetAttributes(&attr, kernel));
    int blocks = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, threads, smem));
    std::printf("%-12s regs=%d local=%zu static-smem=%zu dynamic-smem=%d blocks/SM=%d\n", name,
                attr.numRegs, attr.localSizeBytes, attr.sharedSizeBytes, smem, blocks);
}

// Experimental 64x64 tile: two resident CTAs, one consumer warpgroup each.
__device__ __forceinline__ void small_wgmma_fp8(float* d, uint64_t a, uint64_t b, int accumulate) {
    asm volatile(
        "{ .reg .pred p; setp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1; }\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "l"(a), "l"(b), "r"(accumulate) : "memory");
}

template <bool PROD, int BN>
static __device__ void small_fp8_glu_role(
    __nv_bfloat16* __restrict__ output, const void* map_a, const void* map_gate,
    const void* map_up, const float* __restrict__ activation_scale,
    const float* __restrict__ gate_scale, const float* __restrict__ up_scale,
    unsigned m, unsigned slice, unsigned nblk, void* raw_arena) {
    static_assert(BN == 64 || BN == 128);
    constexpr int NS = BN == 64 ? 4 : 2;
    constexpr int A_BYTES = 64 * 128;
    constexpr int W_BYTES = BN * 128;
    uint64_t* full = static_cast<uint64_t*>(raw_arena);
    uint64_t* empty = full + NS;
    uint8_t* base = static_cast<uint8_t*>(
        sm90_align1024(empty + NS));
    uint8_t* as = base;
    uint8_t* gs = as + NS * A_BYTES;
    uint8_t* us = gs + NS * W_BYTES;
    const int tid = (int)threadIdx.x;

    if constexpr (PROD) {
        if (tid == 0) {
            sm90_tmap_prefetch(map_a);
            sm90_tmap_prefetch(map_gate);
            sm90_tmap_prefetch(map_up);
        }
        if (tid < NS) {
            sm90_mbar_init(full + tid, 1);
            sm90_mbar_init(empty + tid, 1);
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        }
    }
    __syncthreads();

    constexpr int N = 15360;
    constexpr int K = 3840;
    constexpr int TILES_N = N / BN;
    constexpr int KSTEPS = K / 128;
    const int tiles_m = ((int)m + 64 - 1) / 64;
    const int ntiles = tiles_m * TILES_N;

    if constexpr (PROD) {
        if (tid == 0) {
            int issued = 0;
            for (int tile = (int)slice; tile < ntiles; tile += (int)nblk) {
                int tmi, tni;
                sm90_tile_remap(tile, tiles_m, TILES_N, &tmi, &tni);
                const int tm = tmi * 64;
                const int tn = tni * BN;
                for (int ks = 0; ks < KSTEPS; ++ks, ++issued) {
                    const int stage = issued % NS;
                    if (issued >= NS)
                        sm90_mbar_wait(empty + stage,
                                      ((issued / NS) + 1) & 1);
                    sm90_mbar_expect(full + stage, A_BYTES + 2 * W_BYTES);
                    const uint32_t bar = sm90_su32(full + stage);
                    sm90_tma2d(
                        sm90_su32(as + stage * A_BYTES), map_a,
                        ks * 128, tm, bar);
                    sm90_tma2d(
                        sm90_su32(gs + stage * W_BYTES), map_gate,
                        ks * 128, tn, bar);
                    sm90_tma2d(
                        sm90_su32(us + stage * W_BYTES), map_up,
                        ks * 128, tn, bar);
                }
            }
        }
    } else {
        const int local_tid = tid & 127;
        const int warp = local_tid >> 5;
        const int lane = local_tid & 31;
        int step = 0;
        for (int tile = (int)slice; tile < ntiles; tile += (int)nblk) {
            int tmi, tni;
            sm90_tile_remap(tile, tiles_m, TILES_N, &tmi, &tni);
            const int tm = tmi * 64;
            const int tn = tni * BN;
            float gate[BN / 2];
            float up[BN / 2];
            float pg[BN / 2], pu[BN == 64 ? 32 : 1];
#pragma unroll
            for (int i = 0; i < BN / 2; ++i) {
                gate[i] = 0.f;
                up[i] = 0.f;
            }
            for (int ks = 0; ks < KSTEPS; ++ks, ++step) {
                const int stage = step % NS;
                sm90_mbar_wait(full + stage, (step / NS) & 1);
                const uint8_t* a = as + stage * A_BYTES;
                const uint8_t* g = gs + stage * W_BYTES;
                const uint8_t* u = us + stage * W_BYTES;
                if constexpr (BN == 64) {
                    sm90_wg_fence();
#pragma unroll
                    for (int sub = 0; sub < 4; ++sub) {
                        small_wgmma_fp8(pg, sm90_desc(a + sub * 32), sm90_desc(g + sub * 32), sub != 0);
                        small_wgmma_fp8(pu, sm90_desc(a + sub * 32), sm90_desc(u + sub * 32), sub != 0);
                    }
                    sm90_wg_commit();
                    sm90_wg_wait<0>();
#pragma unroll
                    for (int i = 0; i < 32; ++i) {
                        gate[i] += pg[i];
                        up[i] += pu[i];
                    }
                } else {
                    // Reuse the partial so the wider tile fits the 224-register consumer budget.
#pragma unroll
                    for (int matrix = 0; matrix < 2; ++matrix) {
                        sm90_wg_fence();
#pragma unroll
                        for (int sub = 0; sub < 4; ++sub)
                            wgmma_m64n128k32(pg, sm90_desc(a + sub * 32),
                                sm90_desc((matrix == 0 ? g : u) + sub * 32), sub != 0);
                        sm90_wg_commit();
                        sm90_wg_wait<0>();
#pragma unroll
                        for (int i = 0; i < 64; ++i) {
                            if (matrix == 0) gate[i] += pg[i];
                            else up[i] += pg[i];
                        }
                    }
                }
                if (local_tid == 0) sm90_mbar_arrive(empty + stage);
            }

            const int r0 = tm + warp * 16 + (lane >> 2);
            const int c0 = tn + 2 * (lane & 3);
#pragma unroll
            for (int group = 0; group < BN / 8; ++group) {
#pragma unroll
                for (int hi = 0; hi < 2; ++hi) {
                    const int row = r0 + 8 * hi;
                    const int col = c0 + 8 * group;
                    if (row >= (int)m) continue;
                    const float row_scale = activation_scale[row];
#pragma unroll
                    for (int pair = 0; pair < 2; ++pair) {
                        const int ai = 4 * group + 2 * hi + pair;
                        const int column = col + pair;
                        output[(size_t)row * N + column] =
                            pgm90_gemma4_glu_w8a8_epilogue(
                                gate[ai], up[ai], row_scale, gate_scale[column],
                                up_scale[column]);
                    }
                }
            }
        }
    }
    __syncthreads();
    if constexpr (PROD) {
        if (tid < NS) {
            sm90_mbar_inval(full + tid);
            sm90_mbar_inval(empty + tid);
        }
    }
    __syncthreads();
}

template <int BN>
__global__ __maxnreg__(128) void small_fp8_glu(bf16* out, const CUtensorMap* maps,
    const float* as, const float* gs, const float* us, unsigned m) {
    extern __shared__ uint8_t arena[];
    if (threadIdx.x < 128) {
        sm90_reg_dec(32);
        small_fp8_glu_role<true, BN>(out, maps, maps + 1, maps + 2, as, gs, us, m, blockIdx.x, gridDim.x, arena);
    } else {
        sm90_reg_inc(224);
        small_fp8_glu_role<false, BN>(out, maps, maps + 1, maps + 2, as, gs, us, m, blockIdx.x, gridDim.x, arena);
    }
}

__global__ __maxnreg__(160) void promoted_fp8_glu(bf16* out, const CUtensorMap* maps,
    const float* as, const float* gs, const float* us, unsigned m) {
    extern __shared__ uint8_t arena[];
    if (threadIdx.x < 128) {
        sm90_reg_dec(32);
        d_gemm_glu_w8a8_sm90_tma_ws384_gemma4_role<true>(out, maps, maps + 1, maps + 2,
            as, gs, us, m, blockIdx.x, gridDim.x, arena);
    } else {
        sm90_reg_inc(224);
        d_gemm_glu_w8a8_sm90_tma_ws384_gemma4_role<false>(out, maps, maps + 1, maps + 2,
            as, gs, us, m, blockIdx.x, gridDim.x, arena);
    }
}

static CUtensorMap fp8_map(void* base, unsigned rows, unsigned tile_rows) {
    CUtensorMap map{};
    uint64_t dims[] = {3840, rows}, strides[] = {3840};
    uint32_t box[] = {128, tile_rows}, steps[] = {1, 1};
    CKD(cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, base, dims, strides,
        box, steps, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return map;
}

static void check_lt(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cuBLASLt status=%d\n", int(status));
        std::exit(2);
    }
}

struct LtProjection {
    cublasLtHandle_t handle{};
    cublasLtMatmulDesc_t op{};
    cublasLtMatrixLayout_t weight_layout{}, activation_layout{}, output_layout{};
    cublasLtMatmulPreference_t preference{};
    cublasLtMatmulHeuristicResult_t selected{};
    void* workspace{};
    static constexpr size_t workspace_bytes = 32 * 1024 * 1024;

    LtProjection(unsigned m, const float* activation_scale, const float* weight_scale,
                 const void* activation, const void* weight, bf16* output, unsigned n = 15360) {
        check_lt(cublasLtCreate(&handle));
        check_lt(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        cublasOperation_t transpose = CUBLAS_OP_T, normal = CUBLAS_OP_N;
        check_lt(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose)));
        check_lt(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &normal, sizeof(normal)));
        check_lt(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &weight_scale, sizeof(weight_scale)));
        check_lt(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &activation_scale, sizeof(activation_scale)));
        int32_t mode = CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
        int8_t fast_accum = 0;
        check_lt(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof(mode)));
        check_lt(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof(mode)));
        check_lt(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast_accum, sizeof(fast_accum)));
        check_lt(cublasLtMatrixLayoutCreate(&weight_layout, CUDA_R_8F_E4M3, 3840, n, 3840));
        check_lt(cublasLtMatrixLayoutCreate(&activation_layout, CUDA_R_8F_E4M3, 3840, m, 3840));
        check_lt(cublasLtMatrixLayoutCreate(&output_layout, CUDA_R_16BF, n, m, n));
        check_lt(cublasLtMatmulPreferenceCreate(&preference));
        check_lt(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                     &workspace_bytes, sizeof(workspace_bytes)));
        CK(cudaMalloc(&workspace, workspace_bytes));
        cublasLtMatmulHeuristicResult_t choices[32];
        int count = 0, best_index = -1;
        check_lt(cublasLtMatmulAlgoGetHeuristic(handle, op, weight_layout, activation_layout,
            output_layout, output_layout, preference, 32, choices, &count));
        cudaEvent_t begin, end;
        CK(cudaEventCreate(&begin));
        CK(cudaEventCreate(&end));
        float best_ms = INFINITY;
        for (int i = 0; i < count; ++i) {
            if (choices[i].state != CUBLAS_STATUS_SUCCESS) continue;
            selected = choices[i];
            run(activation, weight, output);
            CK(cudaEventRecord(begin));
            for (int repeat = 0; repeat < 5; ++repeat) run(activation, weight, output);
            CK(cudaEventRecord(end));
            CK(cudaEventSynchronize(end));
            float ms;
            CK(cudaEventElapsedTime(&ms, begin, end));
            if (ms < best_ms) { best_ms = ms; best_index = i; }
        }
        CK(cudaEventDestroy(begin));
        CK(cudaEventDestroy(end));
        if (best_index < 0) { std::fprintf(stderr, "No valid vector-scale Lt algorithm\n"); std::exit(2); }
        selected = choices[best_index];
        std::printf("lt_heuristics=%d selected=%d workspace=%zu fast_accum=0 scale=outer_vec_f32\n",
                    count, best_index, selected.workspaceSize);
    }

    void run(const void* activation, const void* weight, bf16* output) {
        const float alpha = 1.f, beta = 0.f;
        check_lt(cublasLtMatmul(handle, op, &alpha, weight, weight_layout, activation,
            activation_layout, &beta, output, output_layout, output, output_layout,
            &selected.algo, workspace, workspace_bytes, nullptr));
    }

    ~LtProjection() {
        cudaFree(workspace);
        cublasLtMatmulPreferenceDestroy(preference);
        cublasLtMatrixLayoutDestroy(weight_layout);
        cublasLtMatrixLayoutDestroy(activation_layout);
        cublasLtMatrixLayoutDestroy(output_layout);
        cublasLtMatmulDescDestroy(op);
        cublasLtDestroy(handle);
    }
};

template <typename T>
static std::vector<T> read_input(const char* directory, const char* name, size_t count) {
    const std::string path = std::string(directory) + "/" + name + ".bin";
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    const size_t bytes = count * sizeof(T);
    if (!file || file.tellg() != std::streamoff(bytes)) {
        std::fprintf(stderr, "Invalid input size: %s expected=%zu\n", path.c_str(), bytes);
        std::exit(2);
    }
    std::vector<T> values(count);
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(values.data()), bytes)) {
        std::fprintf(stderr, "Cannot read input: %s\n", path.c_str());
        std::exit(2);
    }
    return values;
}

template <int BN>
static int fp8_experiment(unsigned m, unsigned seed, const char* input_dir) {
    constexpr unsigned n = 15360, k = 3840;
    constexpr int stages = BN == 64 ? 4 : 2;
    constexpr int small_smem = stages * (8192 + 2 * BN * 128) + 2 * stages * sizeof(uint64_t) + 1024;
    constexpr int control_smem = PGM90_GEMMA4_GLU_ARENA_BYTES;
    CK(cudaFuncSetAttribute(small_fp8_glu<BN>, cudaFuncAttributeMaxDynamicSharedMemorySize, small_smem));
    CK(cudaFuncSetAttribute(promoted_fp8_glu, cudaFuncAttributeMaxDynamicSharedMemorySize, control_smem));
    cudaDeviceProp props{};
    int device;
    CK(cudaGetDevice(&device));
    CK(cudaGetDeviceProperties(&props, device));
    int small_blocks, control_blocks;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&small_blocks, small_fp8_glu<BN>, 256, small_smem));
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&control_blocks, promoted_fp8_glu, 384, control_smem));
    if (!small_blocks || !control_blocks) return 2;
    const int cg = props.multiProcessorCount * control_blocks;
    const int sg = props.multiProcessorCount * small_blocks;
    uint32_t state = 0x12345678u ^ seed * 0x9e3779b9u;
    auto random = [&]() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return float(int32_t(state)) / 2147483648.f;
    };
    auto values = [&](size_t count) {
        std::vector<__nv_fp8_e4m3> v(count);
        for (auto& x : v) x = __nv_fp8_e4m3(random() * 32.f);
        return v;
    };
    auto scales = [&](size_t count) {
        std::vector<float> v(count);
        for (auto& x : v) x = .02f + random() * .005f;
        return v;
    };
    auto a = upload(input_dir ? read_input<__nv_fp8_e4m3>(input_dir, "activation", size_t(m) * k) : values(size_t(m) * k));
    auto g = upload(input_dir ? read_input<__nv_fp8_e4m3>(input_dir, "gate", size_t(n) * k) : values(size_t(n) * k));
    auto u = upload(input_dir ? read_input<__nv_fp8_e4m3>(input_dir, "up", size_t(n) * k) : values(size_t(n) * k));
    auto as = upload(input_dir ? read_input<float>(input_dir, "activation_scale", m) : scales(m));
    auto gs = upload(input_dir ? read_input<float>(input_dir, "gate_scale", n) : scales(n));
    auto us = upload(input_dir ? read_input<float>(input_dir, "up_scale", n) : scales(n));
    std::printf("input_source=%s\n", input_dir ? input_dir : "synthetic");
    auto cm = upload(std::vector<CUtensorMap>{fp8_map(a, m, 128), fp8_map(g, n, 128), fp8_map(u, n, 128)});
    auto sm = upload(std::vector<CUtensorMap>{fp8_map(a, m, 64), fp8_map(g, n, BN), fp8_map(u, n, BN)});
    bf16 *out, *lt_gate, *lt_up;
    CK(cudaMalloc(&out, size_t(m) * n * sizeof(bf16)));
    CK(cudaMalloc(&lt_gate, size_t(m) * n * sizeof(bf16)));
    CK(cudaMalloc(&lt_up, size_t(m) * n * sizeof(bf16)));
    LtProjection gate_plan(m, as, gs, a, g, lt_gate);
    LtProjection up_plan(m, as, us, a, u, lt_up);
    __nv_fp8_e4m3* combined_weight;
    float* combined_scale;
    bf16* combined_projection;
    CK(cudaMalloc(&combined_weight, size_t(2) * n * k));
    CK(cudaMalloc(&combined_scale, size_t(2) * n * sizeof(float)));
    CK(cudaMalloc(&combined_projection, size_t(m) * 2 * n * sizeof(bf16)));
    CK(cudaMemcpy(combined_weight, g, size_t(n) * k, cudaMemcpyDeviceToDevice));
    CK(cudaMemcpy(combined_weight + size_t(n) * k, u, size_t(n) * k, cudaMemcpyDeviceToDevice));
    CK(cudaMemcpy(combined_scale, gs, size_t(n) * sizeof(float), cudaMemcpyDeviceToDevice));
    CK(cudaMemcpy(combined_scale + n, us, size_t(n) * sizeof(float), cudaMemcpyDeviceToDevice));
    LtProjection combined_plan(m, as, combined_scale, a, combined_weight, combined_projection, 2 * n);
    std::printf("cublas_version=%zu\n", cublasLtGetVersion());
    auto launch = [&](int arm) {
        if (arm == 4) {
            combined_plan.run(a, combined_weight, combined_projection);
        } else if (arm == 5) {
            combined_glu<<<props.multiProcessorCount, 256>>>(out, combined_projection, m);
        } else if (arm == 6) {
            gate_plan.run(a, g, lt_gate);
            up_plan.run(a, u, lt_up);
        } else if (arm == 7) {
            shipped_glu<<<props.multiProcessorCount, 256>>>(out, lt_gate, lt_up, size_t(m) * n);
        } else if (arm == 3) {
            combined_plan.run(a, combined_weight, combined_projection);
            combined_glu<<<props.multiProcessorCount, 256>>>(out, combined_projection, m);
        } else if (arm == 2) {
            gate_plan.run(a, g, lt_gate);
            up_plan.run(a, u, lt_up);
            shipped_glu<<<props.multiProcessorCount, 256>>>(out, lt_gate, lt_up, size_t(m) * n);
        } else if (arm == 1) small_fp8_glu<BN><<<sg, 256, small_smem>>>(out, sm, as, gs, us, m);
        else promoted_fp8_glu<<<cg, 384, control_smem>>>(out, cm, as, gs, us, m);
        CK(cudaGetLastError());
    };
    std::vector<bf16> ref(size_t(m) * n), got(ref.size());
    launch(false);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(ref.data(), out, ref.size() * sizeof(bf16), cudaMemcpyDeviceToHost));
    launch(true);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(got.data(), out, got.size() * sizeof(bf16), cudaMemcpyDeviceToHost));
    size_t mismatches = 0, nonfinite = 0;
    double err2 = 0, ref2 = 0;
    float maxerr = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const float r = __bfloat162float(ref[i]), x = __bfloat162float(got[i]);
        nonfinite += !std::isfinite(r) || !std::isfinite(x);
        mismatches += std::memcmp(&ref[i], &got[i], sizeof(bf16)) != 0;
        const double e = double(x) - r;
        err2 += e * e;
        ref2 += double(r) * r;
        maxerr = std::max(maxerr, float(std::abs(e)));
    }
    launch(2);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(got.data(), out, got.size() * sizeof(bf16), cudaMemcpyDeviceToHost));
    double lt_err2 = 0;
    size_t lt_nonfinite = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const float value = __bfloat162float(got[i]);
        lt_nonfinite += !std::isfinite(value);
        const double delta = double(value) - __bfloat162float(ref[i]);
        lt_err2 += delta * delta;
    }
    std::printf("lt_vs_promoted_rel_L2=%.9g lt_nonfinite=%zu\n",
                std::sqrt(lt_err2 / std::max(ref2, 1e-30)), lt_nonfinite);
    std::vector<bf16> combined(got.size());
    launch(3);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(combined.data(), out, combined.size() * sizeof(bf16), cudaMemcpyDeviceToHost));
    size_t combined_mismatches = 0, combined_nonfinite = 0;
    double combined_err2 = 0, lt_ref2 = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const float x = __bfloat162float(combined[i]), y = __bfloat162float(got[i]);
        combined_nonfinite += !std::isfinite(x);
        combined_mismatches += std::memcmp(&combined[i], &got[i], sizeof(bf16)) != 0;
        const double delta = double(x) - y;
        combined_err2 += delta * delta;
        lt_ref2 += double(y) * y;
    }
    std::printf("combined_vs_pair_mismatches=%zu combined_nonfinite=%zu combined_vs_pair_rel_L2=%.9g\n",
                combined_mismatches, combined_nonfinite, std::sqrt(combined_err2 / std::max(lt_ref2, 1e-30)));
    cudaEvent_t begin, end;
    CK(cudaEventCreate(&begin));
    CK(cudaEventCreate(&end));
    auto measure = [&](int arm) {
        CK(cudaEventRecord(begin));
        launch(arm);
        CK(cudaEventRecord(end));
        CK(cudaEventSynchronize(end));
        float ms;
        CK(cudaEventElapsedTime(&ms, begin, end));
        return ms;
    };
    for (int i = 0; i < 4; ++i)
        for (int arm = 0; arm < 4; ++arm) measure(arm);
    std::vector<float> ctl, candidate, lt, joined;
    for (int i = 0; i < 24; ++i) {
        float times[4];
        // Balanced positions in forward and reverse cyclic orders.
        const int orders[8][4] = {{0,1,2,3}, {1,2,3,0}, {2,3,0,1}, {3,0,1,2},
                                  {3,2,1,0}, {0,3,2,1}, {1,0,3,2}, {2,1,0,3}};
        for (int arm : orders[i % 8]) times[arm] = measure(arm);
        ctl.push_back(times[0]);
        candidate.push_back(times[1]);
        lt.push_back(times[2]);
        joined.push_back(times[3]);
        std::printf("round=%d control_ms=%.6f candidate_ms=%.6f lt_ms=%.6f combined_ms=%.6f\n",
                    i, times[0], times[1], times[2], times[3]);
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
    std::sort(ctl.begin(), ctl.end());
    std::sort(candidate.begin(), candidate.end());
    std::sort(lt.begin(), lt.end());
    std::sort(joined.begin(), joined.end());
    const float control_median = (ctl[11] + ctl[12]) / 2;
    const float candidate_median = (candidate[11] + candidate[12]) / 2;
    const float lt_median = (lt[11] + lt[12]) / 2;
    const float combined_median = (joined[11] + joined[12]) / 2;
    std::printf("combined_pair_glu_p50_ms=%.6f combined_vs_pair_speedup=%.6f\n",
                combined_median, lt_median / combined_median);
    std::printf("lt_pair_glu_p50_ms=%.6f candidate_vs_lt_speedup=%.6f\n",
                lt_median, lt_median / candidate_median);
    std::printf("FP8 M%u N%u K%u seed%u control=promoted-production-128x128 candidate=promoted-64x%d stages=%d grids=%d/%d\n",m,n,k,seed,BN,stages,cg,sg);
    print_resources("control", reinterpret_cast<const void*>(promoted_fp8_glu),384,control_smem);
    print_resources("candidate", reinterpret_cast<const void*>(small_fp8_glu<BN>),256,small_smem);
    std::printf("mismatches=%zu nonfinite=%zu rel_L2=%.9g max_abs=%.9g control_p50_ms=%.6f candidate_p50_ms=%.6f speedup=%.6f\n",
        mismatches,nonfinite,std::sqrt(err2/std::max(ref2,1e-30)),maxerr,control_median,candidate_median,control_median/candidate_median);
    std::vector<float> components[4];
    for (int i = 0; i < 20; ++i) {
        for (int offset = 0; offset < 4; ++offset) {
            const int arm = (i + offset) % 4;
            const float ms = measure(arm + 4);
            if (i >= 4) components[arm].push_back(ms);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
    const char* names[] = {"combined_gemm", "combined_glu", "pair_gemm", "pair_glu"};
    for (int arm = 0; arm < 4; ++arm) {
        auto& times = components[arm];
        std::sort(times.begin(), times.end());
        std::printf("component=%s p50_ms=%.6f\n", names[arm], (times[7] + times[8]) / 2);
    }
    CK(cudaEventDestroy(begin));
    CK(cudaEventDestroy(end));
    const bool quant_correct = compare_glu_quant(out, lt_gate, lt_up, m, props.multiProcessorCount * 2);
    cudaFree(a);
    cudaFree(g);
    cudaFree(u);
    cudaFree(as);
    cudaFree(gs);
    cudaFree(us);
    cudaFree(cm);
    cudaFree(sm);
    cudaFree(out);
    cudaFree(lt_gate);
    cudaFree(lt_up);
    cudaFree(combined_weight);
    cudaFree(combined_scale);
    cudaFree(combined_projection);
    return nonfinite == 0 && lt_nonfinite == 0 && mismatches == 0 &&
           combined_nonfinite == 0 && combined_mismatches == 0 && quant_correct ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc >= 2 && (std::strcmp(argv[1], "--fp8") == 0 ||
                      std::strcmp(argv[1], "--fp8-wide") == 0)) {
        const unsigned m = argc >= 3 ? std::strtoul(argv[2], nullptr, 10) : 4096;
        const unsigned seed = argc >= 4 ? std::strtoul(argv[3], nullptr, 10) : 0;
        if (argc > 5 || (m != 4096 && m != 8192)) return 2;
        const char* input_dir = argc == 5 ? argv[4] : nullptr;
        return std::strcmp(argv[1], "--fp8-wide") == 0
            ? fp8_experiment<128>(m, seed, input_dir) : fp8_experiment<64>(m, seed, input_dir);
    }
    if (argc > 3) {
        std::fprintf(stderr, "usage: %s [--fp8|--fp8-wide] [4096|8192] [seed] [input-dir (FP8 only)]\n", argv[0]);
        return 2;
    }
    const unsigned m = argc >= 2 ? std::strtoul(argv[1], nullptr, 10) : 4096;
    const unsigned seed = argc == 3 ? std::strtoul(argv[2], nullptr, 10) : 0;
    if (m != 4096 && m != 8192) return 2;
    constexpr unsigned n = 15360;
    constexpr unsigned k = 3840;
    constexpr int grid = 132;
    const int baseline_smem = 2 * PGM90_WS384_ARENA;
    CK(cudaFuncSetAttribute(fused_glu, cudaFuncAttributeMaxDynamicSharedMemorySize, GLU_SMEM));
    CK(cudaFuncSetAttribute(shipped_ws384, cudaFuncAttributeMaxDynamicSharedMemorySize,
                            baseline_smem));

    uint32_t state = 0x12345678u ^ seed * 0x9e3779b9u;
    auto make_values = [&](size_t count) {
        std::vector<bf16> values(count);
        for (bf16& value : values) {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            const float x = float(int32_t(state)) * (1.0f / 2147483648.0f) * 0.025f;
            value = __float2bfloat16(x);
        }
        return values;
    };
    bf16* a = upload(make_values(size_t(m) * k));
    bf16* wg = upload(make_values(size_t(n) * k));
    bf16* wu = upload(make_values(size_t(n) * k));
    bf16 *out, *gate, *up;
    CK(cudaMalloc(&out, size_t(m) * n * sizeof(bf16)));
    CK(cudaMalloc(&gate, size_t(m) * n * sizeof(bf16)));
    CK(cudaMalloc(&up, size_t(m) * n * sizeof(bf16)));
    CUtensorMap* maps = upload(std::vector<CUtensorMap>{make_map(a, m, k), make_map(wg, n, k),
                                                        make_map(wu, n, k)});

    shipped_ws384<<<grid, 384, baseline_smem>>>(gate, maps, maps + 1, m, n, k);
    shipped_ws384<<<grid, 384, baseline_smem>>>(up, maps, maps + 2, m, n, k);
    shipped_glu<<<grid, 256>>>(out, gate, up, size_t(m) * n);
    CK(cudaDeviceSynchronize());
    std::vector<bf16> reference(size_t(m) * n);
    CK(cudaMemcpy(reference.data(), out, reference.size() * sizeof(bf16), cudaMemcpyDeviceToHost));

    fused_glu<<<grid, 384, GLU_SMEM>>>(out, maps, maps + 1, maps + 2, m, n, k);
    CK(cudaDeviceSynchronize());
    std::vector<bf16> actual(size_t(m) * n);
    CK(cudaMemcpy(actual.data(), out, actual.size() * sizeof(bf16), cudaMemcpyDeviceToHost));
    size_t mismatches = 0;
    double err2 = 0.0, ref2 = 0.0;
    float max_error = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        const float got = __bfloat162float(actual[i]);
        const float ref = __bfloat162float(reference[i]);
        mismatches += std::memcmp(&actual[i], &reference[i], sizeof(bf16)) != 0;
        const float error = got - ref;
        err2 += double(error) * error;
        ref2 += double(ref) * ref;
        max_error = std::max(max_error, std::abs(error));
    }
    const double rel_l2 = std::sqrt(err2 / std::max(ref2, 1e-30));

    for (int warmup = 0; warmup < 2; ++warmup) {
        (void)timed(false, 1, grid, GLU_SMEM, out, gate, up, maps, m, n, k);
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
        (void)timed(true, 1, grid, GLU_SMEM, out, gate, up, maps, m, n, k);
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
    std::vector<float> control, candidate;
    for (int repeat = 0; repeat < 15; ++repeat) {
        if ((repeat & 1) == 0) {
            control.push_back(timed(false, 1, grid, GLU_SMEM, out, gate, up, maps, m, n, k));
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
            candidate.push_back(timed(true, 1, grid, GLU_SMEM, out, gate, up, maps, m, n, k));
        } else {
            candidate.push_back(timed(true, 1, grid, GLU_SMEM, out, gate, up, maps, m, n, k));
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
            control.push_back(timed(false, 1, grid, GLU_SMEM, out, gate, up, maps, m, n, k));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
    std::sort(control.begin(), control.end());
    std::sort(candidate.begin(), candidate.end());

    std::printf(
        "Gemma-4 BF16 fused gate+up+GeGLU M%u N%u K%u candidate-BN128 control-BN256 "
        "NS%d seed%u\n",
        m, n, k, PLOW_GLU_NS, seed);
    print_resources("control GEMM", reinterpret_cast<const void*>(shipped_ws384), 384,
                    baseline_smem);
    print_resources("fused GLU", reinterpret_cast<const void*>(fused_glu), 384, GLU_SMEM);
    std::printf("correctness mismatches=%zu rel-L2=%.3e max-abs=%.6g %s\n", mismatches, rel_l2,
                max_error, mismatches == 0 ? "PASS" : "FAIL");
    std::printf("control  min=%.3f ms p50=%.3f ms\n", control.front(), control[7]);
    std::printf("candidate min=%.3f ms p50=%.3f ms speedup=%.3fx\n", candidate.front(),
                candidate[7], control.front() / candidate.front());

    cudaFree(a);
    cudaFree(wg);
    cudaFree(wu);
    cudaFree(out);
    cudaFree(gate);
    cudaFree(up);
    cudaFree(maps);
    return mismatches == 0 ? 0 : 1;
}
