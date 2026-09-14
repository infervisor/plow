// Exact Gemma-4 gate/up projection experiment. The control is two shipped WS384 BF16 GEMMs
// followed by d_glu; the candidate shares A staging and fuses the same BF16-boundary GeGLU.
//
// Build NS=3/4 independently:
//   nvcc -std=c++17 -gencode arch=compute_90a,code=sm_90a -O3 \
//     -I runtime/common -I runtime/nvidia -include cstdint -Xptxas=-v \
//     -DPLOW_GLU_NS=3 runtime/nvidia/experiments/gemma4_ws384_fused_glu.cu \
//     -lcuda -o /tmp/gemma4_ws384_glu_ns3
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
#define PLOW_NV_W8A8 0
#define PGM90_UNI_BN256 1
#define PLOW_NV_SEG_GEMM 1
#define PLOW_NV_SEGMENTS 1
#define PLOW_NV_SEG_WS384 1
#define PGM90_WS384_BN 256
#define PGM90_UNI256_NS 3
#define PGM90_WS384_SMEPI 0
#define PGM90_WS384_PREFETCH 1
#define PGM90_WS384_ISSUE_CURSOR 1
#define PGM_ARENA_BF16 (128 * 1024)
#define PLOW_NV_GEMMA 1

#include "sm120_common.cuh"
#include "op_gemm_sm90.cuh"
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

int main(int argc, char** argv) {
    if (argc > 3) {
        std::fprintf(stderr, "usage: %s [4096|8192] [seed]\n", argv[0]);
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
