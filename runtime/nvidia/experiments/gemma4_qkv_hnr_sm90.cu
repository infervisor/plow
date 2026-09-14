// Exact Gemma-4 sliding-attention projection + HeadNorm/RoPE experiment.
// The candidate materializes each WGMMA accumulator through BF16 in shared memory,
// then applies the production HD256 warp reduction and packed KV address calculation.
//
// Build:
//   /usr/local/cuda/bin/nvcc -std=c++17 -gencode arch=compute_90a,code=sm_90a -O3 \
//     -I runtime/common -I runtime/nvidia -include cstdint -Xptxas=-v \
//     runtime/nvidia/experiments/gemma4_qkv_hnr_sm90.cu -lcuda \
//     -o /tmp/gemma4_qkv_hnr_sm90
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#define PLOW_NV_HOPPER 1
#define PLOW_NV_PREFILL 1
#define PLOW_NV_THREADS 256u
#define PLOW_NV_TMA_GEMM 1
#define PLOW_NV_W8A8 0
#define PGM90_UNI_BN256 1
#define PLOW_NV_SEG_GEMM 1
#define PLOW_NV_SEGMENTS 1
#define PLOW_NV_SEG_WS384 1
#define PGM90_WS384_BN 256
#define PGM90_UNI256_NS 4
#define PGM90_WS384_SMEPI 0
#define PGM90_WS384_PREFETCH 1
#define PGM90_WS384_ISSUE_CURSOR 1
#define PGM_ARENA_BF16 (128 * 1024)
#define PLOW_NV_GEMMA 1

#include "sm120_common.cuh"
#include "op_gemm_sm90.cuh"
#include "op_norm.cuh"

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

constexpr unsigned HD = 256;
constexpr unsigned K = 3840;
constexpr unsigned GRID = 132;
constexpr int EXTRA_SMEM = 32 * HD * sizeof(bf16);
constexpr int CONTROL_SMEM = 2 * PGM90_WS384_ARENA;
constexpr int FUSED_SMEM = CONTROL_SMEM + EXTRA_SMEM;

static_assert(PGM90_WS384_BN == HD);
static_assert(FUSED_SMEM <= 227328, "H100 opt-in shared-memory limit exceeded");

__device__ __forceinline__ void fused_hnr_barrier(int consumer) {
    asm volatile("bar.sync %0, %1;" ::"r"(consumer + 1), "r"(128) : "memory");
}

__device__ __forceinline__ size_t fused_hnr_obase(
    unsigned row, unsigned head, unsigned nhead, unsigned out_row0, unsigned out_stride,
    unsigned kv_mask, const int* __restrict__ pos, const int* __restrict__ pfslot) {
    if (!out_stride) return ((size_t)(out_row0 + row) * nhead + head) * HD;
    if (pfslot)
        return ((size_t)((unsigned)pfslot[row] * nhead + head) * out_stride +
                ((unsigned)pos[row] & kv_mask)) *
               HD;
    return ((size_t)head * out_stride + ((out_row0 + row) & kv_mask)) * HD;
}

// This is the HD256 pack-of-four arm of d_headnorm_rope. The staged input is already
// BF16, so the GEMM-to-HNR numerical boundary is identical to the split control.
__device__ __forceinline__ void fused_hnr_row(
    bf16* __restrict__ out, const bf16* __restrict__ staged,
    const bf16* __restrict__ gamma, const float* __restrict__ cosb,
    const float* __restrict__ sinb, const int* __restrict__ pos,
    const int* __restrict__ pfslot, unsigned row, unsigned head, unsigned nhead,
    unsigned out_row0, unsigned out_stride, unsigned kv_mask, unsigned skip_norm, float eps) {
    const unsigned lane = threadIdx.x & 31u;
    constexpr unsigned C = HD / 128;
    constexpr unsigned CH = C / 2;
    float v[4 * C], g[4 * C];
#pragma unroll
    for (unsigned c = 0; c < C; ++c) {
        const ushort4 xv = *(const ushort4*)(staged + 4u * (lane + 32u * c));
        const unsigned short* xs = (const unsigned short*)&xv;
#pragma unroll
        for (int j = 0; j < 4; ++j)
            v[4 * c + j] = __bfloat162float(*(const bf16*)&xs[j]);
        if (gamma) {
            const ushort4 gv = *(const ushort4*)(gamma + 4u * (lane + 32u * c));
            const unsigned short* gs = (const unsigned short*)&gv;
#pragma unroll
            for (int j = 0; j < 4; ++j)
                g[4 * c + j] = norm_weight(__bfloat162float(*(const bf16*)&gs[j]));
        } else {
#pragma unroll
            for (int j = 0; j < 4; ++j) g[4 * c + j] = 1.0f;
        }
    }
    float inv = 1.0f;
    if (!skip_norm) {
        float ss = 0.0f;
#pragma unroll
        for (unsigned e = 0; e < 4 * C; ++e) ss += v[e] * v[e];
        inv = rsqrtf(warp_sum32(ss) * __fdividef(1.0f, (float)HD) + eps);
    }
#pragma unroll
    for (unsigned e = 0; e < 4 * C; ++e) v[e] = gemma3_bf16_round(v[e] * inv * g[e]);

    if (cosb) {
#if (PLOW_NV_GEMMA && PLOW_NV_GEMMA_HNR_BF16) || PLOW_NV_GEMMA3
#pragma unroll
        for (unsigned e = 0; e < 4 * C; ++e)
            v[e] = __bfloat162float(__float2bfloat16(v[e]));
#endif
        const size_t p = (size_t)pos[row] * (HD / 2);
        float r[4 * C];
#pragma unroll
        for (unsigned c = 0; c < CH; ++c) {
            const size_t j = p + 4u * (lane + 32u * c);
            const float4 cc = *(const float4*)(cosb + j);
            const float4 sv = *(const float4*)(sinb + j);
            const float* cp = (const float*)&cc;
            const float* sp = (const float*)&sv;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const unsigned lo = 4 * c + k, hi = 4 * (c + CH) + k;
#if (PLOW_NV_GEMMA && PLOW_NV_GEMMA_HNR_BF16) || PLOW_NV_GEMMA3
                const float cr = __bfloat162float(__float2bfloat16(cp[k]));
                const float sr = __bfloat162float(__float2bfloat16(sp[k]));
                r[lo] = v[lo] * cr - v[hi] * sr;
                r[hi] = v[hi] * cr + v[lo] * sr;
#else
                r[lo] = v[lo] * cp[k] - v[hi] * sp[k];
                r[hi] = v[hi] * cp[k] + v[lo] * sp[k];
#endif
            }
        }
#pragma unroll
        for (unsigned e = 0; e < 4 * C; ++e) v[e] = r[e];
    }

    const size_t obase =
        fused_hnr_obase(row, head, nhead, out_row0, out_stride, kv_mask, pos, pfslot);
#pragma unroll
    for (unsigned c = 0; c < C; ++c) {
        ushort4 packed;
        unsigned short* dst = (unsigned short*)&packed;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const bf16 value = __float2bfloat16(v[4 * c + k]);
            dst[k] = *(const unsigned short*)&value;
        }
        *(ushort4*)(out + obase + 4u * (lane + 32u * c)) = packed;
    }
}

template <bool Producer>
__device__ void fused_projection_hnr_body(
    bf16* __restrict__ out, const void* map_a, const void* map_b,
    const bf16* __restrict__ gamma, const float* __restrict__ cosb,
    const float* __restrict__ sinb, const int* __restrict__ pos,
    const int* __restrict__ pfslot, unsigned m, unsigned n, unsigned nhead,
    unsigned out_row0, unsigned out_stride, unsigned kv_mask, unsigned skip_norm,
    float eps, unsigned slice, unsigned nblk, bf16* arena) {
    constexpr int NS = PGM90_UNI256_NS;
    constexpr int BKB = 128;
    uint64_t* full = (uint64_t*)arena;
    uint64_t* empty = full + NS;
    uint8_t* base = (uint8_t*)sm90_align1024(arena + PGM90_U256_MBAR_BF16);
    uint8_t* as = base;
    uint8_t* bs = base + NS * PGM90_A8BUF;
    bf16* extra = (bf16*)(bs + NS * PGM90_WS384_BBUF);
    const int tid = threadIdx.x;

    if constexpr (Producer) {
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

    const int tiles_m = ((int)m + PGM90_BM - 1) / PGM90_BM;
    const int tiles_n = ((int)n + PGM90_WS384_BN - 1) / PGM90_WS384_BN;
    const int ntiles = tiles_m * tiles_n;
    constexpr int ksteps = K / 64;

    if constexpr (Producer) {
        if (tid == 0) {
            int issued = 0;
            for (int tile = (int)slice; tile < ntiles; tile += (int)nblk) {
                int tmi, tni;
                sm90_tile_remap(tile, tiles_m, tiles_n, &tmi, &tni);
                const int tm = tmi * PGM90_BM;
                const int tn = tni * PGM90_WS384_BN;
                for (int ks = 0; ks < ksteps; ++ks, ++issued) {
                    const int stage = issued % NS;
                    if (issued >= NS) sm90_mbar_wait(empty + stage, ((issued / NS) + 1) & 1);
                    sm90_mbar_expect(full + stage, PGM90_WS384_TXB);
                    const uint32_t bar = sm90_su32(full + stage);
                    sm90_tma2d(sm90_su32(as + stage * PGM90_A8BUF), map_a, ks * 64, tm, bar);
                    uint8_t* weight = bs + stage * PGM90_WS384_BBUF;
                    sm90_tma2d(sm90_su32(weight), map_b, ks * 64, tn, bar);
                    sm90_tma2d(sm90_su32(weight + 128 * BKB), map_b, ks * 64, tn + 128,
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
        for (int tile = (int)slice; tile < ntiles; tile += (int)nblk) {
            int tmi, tni;
            sm90_tile_remap(tile, tiles_m, tiles_n, &tmi, &tni);
            const int tm = tmi * PGM90_BM;
            const int tn = tni * PGM90_WS384_BN;
            const int head = tn / (int)HD;
            int previous = -1;
            {
                float acc[PGM90_WS384_BN / 2];
                for (int ks = 0; ks < ksteps; ++ks, ++step) {
                    const int stage = step % NS;
                    sm90_mbar_wait(full + stage, (step / NS) & 1);
                    const uint8_t* activation =
                        as + stage * PGM90_A8BUF + consumer * PGM90_MSLAB * BKB;
                    const uint8_t* weight = bs + stage * PGM90_WS384_BBUF;
                    sm90_wg_fence();
#pragma unroll
                    for (int sub = 0; sub < 4; ++sub) {
                        const int accumulate = (ks == 0 && sub == 0) ? 0 : 1;
                        wgmma_m64n256k16(acc, sm90_desc(activation + sub * 32),
                                         sm90_desc(weight + sub * 32), accumulate);
                    }
                    sm90_wg_commit();
                    sm90_wg_wait<1>();
                    if (previous >= 0 && lt == 0) sm90_mbar_arrive(empty + previous);
                    previous = stage;
                }
                sm90_wg_wait<0>();
                asm volatile("bar.sync 3, 256;" ::: "memory");

                const int local_row0 = warp * 16 + (lane >> 2);
                const int local_col0 = 2 * (lane & 3);
#pragma unroll
                for (int group = 0; group < PGM90_WS384_BN / 8; ++group) {
#pragma unroll
                    for (int hi = 0; hi < 2; ++hi) {
                        const int local_row = local_row0 + 8 * hi;
                        const int local_col = local_col0 + 8 * group;
                        bf16* staged = consumer == 0
                                           ? (bf16*)(bs + previous * PGM90_WS384_BBUF) +
                                                 (size_t)local_row * HD
                                           : local_row < 32
                                                 ? (bf16*)(as + previous * PGM90_A8BUF) +
                                                       (size_t)local_row * HD
                                                 : extra + (size_t)(local_row - 32) * HD;
                        *(__nv_bfloat162*)(staged + local_col) = __floats2bfloat162_rn(
                            acc[4 * group + 2 * hi], acc[4 * group + 2 * hi + 1]);
                    }
                }
                fused_hnr_barrier(consumer);
            }

            for (int local_row = warp; local_row < 64; local_row += 4) {
                const unsigned row = tm + consumer * 64 + local_row;
                if (row >= m) continue;
                const bf16* staged =
                    consumer == 0
                        ? (const bf16*)(bs + previous * PGM90_WS384_BBUF) +
                              (size_t)local_row * HD
                        : local_row < 32
                              ? (const bf16*)(as + previous * PGM90_A8BUF) +
                                    (size_t)local_row * HD
                              : extra + (size_t)(local_row - 32) * HD;
                fused_hnr_row(out, staged, gamma, cosb, sinb, pos, pfslot, row, head,
                              nhead, out_row0, out_stride, kv_mask, skip_norm, eps);
            }
            fused_hnr_barrier(consumer);
            if (previous >= 0 && lt == 0) sm90_mbar_arrive(empty + previous);
        }
    }
    __syncthreads();
    if constexpr (Producer) {
        if (tid < NS) {
            sm90_mbar_inval(full + tid);
            sm90_mbar_inval(empty + tid);
        }
    }
    __syncthreads();
}

__global__ __maxnreg__(160) void fused_projection_hnr(
    bf16* out, const void* map_a, const void* map_b, const bf16* gamma,
    const float* cosb, const float* sinb, const int* pos, const int* pfslot,
    unsigned m, unsigned n, unsigned nhead, unsigned out_row0, unsigned out_stride,
    unsigned kv_mask, unsigned skip_norm, float eps) {
    extern __shared__ uint8_t raw[];
    if (threadIdx.x < 128) {
        sm90_reg_dec(32);
        fused_projection_hnr_body<true>(out, map_a, map_b, gamma, cosb, sinb, pos,
                                         pfslot, m, n, nhead, out_row0, out_stride,
                                         kv_mask, skip_norm, eps, blockIdx.x, gridDim.x,
                                         (bf16*)raw);
    } else {
        sm90_reg_inc(224);
        fused_projection_hnr_body<false>(out, map_a, map_b, gamma, cosb, sinb, pos,
                                          pfslot, m, n, nhead, out_row0, out_stride,
                                          kv_mask, skip_norm, eps, blockIdx.x, gridDim.x,
                                          (bf16*)raw);
    }
}

__global__ __maxnreg__(160) void control_projection(bf16* out, const void* map_a,
                                                     const void* map_b, unsigned m,
                                                     unsigned n) {
    extern __shared__ uint8_t raw[];
    if (threadIdx.x < 128) {
        sm90_reg_dec(32);
        d_gemm_sm90_tma_ws384_role<true, false>(out, map_a, map_b, nullptr, nullptr,
                                                m, n, K, 0, blockIdx.x, gridDim.x,
                                                (bf16*)raw);
    } else {
        sm90_reg_inc(224);
        d_gemm_sm90_tma_ws384_role<false, false>(out, map_a, map_b, nullptr, nullptr,
                                                 m, n, K, 0, blockIdx.x, gridDim.x,
                                                 (bf16*)raw);
    }
}

__global__ void control_hnr(bf16* out, const bf16* x, const bf16* gamma,
                            const float* cosb, const float* sinb, const int* pos,
                            const int* pfslot, unsigned m, unsigned nhead,
                            unsigned out_row0, unsigned out_stride, unsigned kv_mask,
                            unsigned skip_norm, float eps) {
    d_headnorm_rope<HD>(out, x, gamma, cosb, sinb, pos, m, nhead, eps, out_row0,
                           out_stride, kv_mask, skip_norm, blockIdx.x, gridDim.x, 0,
                           pfslot);
}

static CUtensorMap make_map(void* base, unsigned rows, unsigned cols) {
    CUtensorMap map{};
    uint64_t dims[] = {cols, rows};
    uint64_t strides[] = {uint64_t(cols) * sizeof(bf16)};
    uint32_t box[] = {64, 128};
    uint32_t steps[] = {1, 1};
    CKD(cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, base,
                               dims, strides, box, steps,
                               CU_TENSOR_MAP_INTERLEAVE_NONE,
                               CU_TENSOR_MAP_SWIZZLE_128B,
                               CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return map;
}

template <typename T>
static T* upload(const std::vector<T>& host) {
    T* device = nullptr;
    CK(cudaMalloc(&device, host.size() * sizeof(T)));
    CK(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    return device;
}

struct Projection {
    unsigned n;
    unsigned nhead;
    bf16* weight;
    bf16* raw;
    bf16* control;
    bf16* candidate;
    const bf16* gamma;
    const float* cosb;
    const float* sinb;
    unsigned out_stride;
    unsigned kv_mask;
    unsigned skip_norm;
    CUtensorMap* map;
};

static void launch_control(const Projection& p, const CUtensorMap* map_a,
                           const int* pos, const int* pfslot, unsigned m) {
    control_projection<<<GRID, 384, CONTROL_SMEM>>>(p.raw, map_a, p.map, m, p.n);
    control_hnr<<<GRID, 256>>>(p.control, p.raw, p.gamma, p.cosb, p.sinb, pos,
                               p.out_stride ? pfslot : nullptr, m, p.nhead, 0,
                               p.out_stride, p.kv_mask, p.skip_norm, 1e-6f);
}

static void launch_candidate(const Projection& p, const CUtensorMap* map_a,
                             const int* pos, const int* pfslot, unsigned m) {
    fused_projection_hnr<<<GRID, 384, FUSED_SMEM>>>(
        p.candidate, map_a, p.map, p.gamma, p.cosb, p.sinb, pos,
        p.out_stride ? pfslot : nullptr, m, p.n, p.nhead, 0, p.out_stride,
        p.kv_mask, p.skip_norm, 1e-6f);
}

template <class Launch>
static float timed(Launch launch) {
    cudaEvent_t begin, end;
    CK(cudaEventCreate(&begin));
    CK(cudaEventCreate(&end));
    CK(cudaEventRecord(begin));
    launch();
    CK(cudaEventRecord(end));
    CK(cudaEventSynchronize(end));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, begin, end));
    CK(cudaEventDestroy(begin));
    CK(cudaEventDestroy(end));
    return ms;
}

static void print_resources(const char* name, const void* kernel, int threads, int smem) {
    cudaFuncAttributes attr{};
    CK(cudaFuncGetAttributes(&attr, kernel));
    int blocks = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, threads, smem));
    std::printf("%-20s regs=%d local=%zu static-smem=%zu dynamic-smem=%d blocks/SM=%d\n",
                name, attr.numRegs, attr.localSizeBytes, attr.sharedSizeBytes, smem, blocks);
}

int main(int argc, char** argv) {
    if (argc > 3) {
        std::fprintf(stderr, "usage: %s [4096|8192] [seed]\n", argv[0]);
        return 2;
    }
    const unsigned m = argc >= 2 ? std::strtoul(argv[1], nullptr, 10) : 4096;
    const unsigned seed_arg = argc == 3 ? std::strtoul(argv[2], nullptr, 10) : 1;
    if (m != 4096 && m != 8192) return 2;

    CK(cudaFuncSetAttribute(control_projection,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, CONTROL_SMEM));
    CK(cudaFuncSetAttribute(fused_projection_hnr,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, FUSED_SMEM));

    uint32_t state = 0x12345678u ^ seed_arg * 0x9e3779b9u;
    auto values = [&](size_t count, float scale) {
        std::vector<bf16> host(count);
        for (bf16& value : host) {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            value = __float2bfloat16(float(int32_t(state)) *
                                     (1.0f / 2147483648.0f) * scale);
        }
        return host;
    };

    bf16* activation = upload(values((size_t)m * K, 0.025f));
    bf16* wq = upload(values((size_t)4096 * K, 0.025f));
    bf16* wk = upload(values((size_t)2048 * K, 0.025f));
    bf16* wv = upload(values((size_t)2048 * K, 0.025f));
    bf16* qgamma = upload(values(HD, 0.25f));
    bf16* kgamma = upload(values(HD, 0.25f));

    const unsigned slot_rows = m / 2;
    std::vector<int> hpos(m), hslot(m);
    for (unsigned row = 0; row < m; ++row) {
        hslot[row] = row & 1u;
        hpos[row] = row >> 1;
    }
    int* pos = upload(hpos);
    int* pfslot = upload(hslot);
    std::vector<float> hcos((size_t)slot_rows * HD / 2);
    std::vector<float> hsin(hcos.size());
    for (unsigned row = 0; row < slot_rows; ++row)
        for (unsigned col = 0; col < HD / 2; ++col) {
            const float angle = float(row) * std::pow(10000.0f, -2.0f * col / HD);
            hcos[(size_t)row * HD / 2 + col] = std::cos(angle);
            hsin[(size_t)row * HD / 2 + col] = std::sin(angle);
        }
    float* cosb = upload(hcos);
    float* sinb = upload(hsin);

    CUtensorMap host_maps[] = {make_map(activation, m, K), make_map(wq, 4096, K),
                               make_map(wk, 2048, K), make_map(wv, 2048, K)};
    CUtensorMap* maps = upload(std::vector<CUtensorMap>(host_maps, host_maps + 4));

    auto allocate = [&](unsigned n, unsigned nhead, bool kv, bf16* weight,
                        const bf16* gamma, bool rope, unsigned skip, CUtensorMap* map) {
        Projection p{};
        p.n = n;
        p.nhead = nhead;
        p.weight = weight;
        p.gamma = gamma;
        p.cosb = rope ? cosb : nullptr;
        p.sinb = rope ? sinb : nullptr;
        p.out_stride = kv ? slot_rows : 0;
        p.kv_mask = kv ? slot_rows - 1 : 0;
        p.skip_norm = skip;
        p.map = map;
        const size_t elements = kv ? (size_t)2 * nhead * slot_rows * HD : (size_t)m * n;
        CK(cudaMalloc(&p.raw, (size_t)m * n * sizeof(bf16)));
        CK(cudaMalloc(&p.control, elements * sizeof(bf16)));
        CK(cudaMalloc(&p.candidate, elements * sizeof(bf16)));
        CK(cudaMemset(p.control, 0, elements * sizeof(bf16)));
        CK(cudaMemset(p.candidate, 0, elements * sizeof(bf16)));
        return p;
    };
    Projection q = allocate(4096, 16, false, wq, qgamma, true, 0, maps + 1);
    Projection k = allocate(2048, 8, true, wk, kgamma, true, 0, maps + 2);
    Projection v = allocate(2048, 8, true, wv, nullptr, false, 1, maps + 3);
    std::vector<Projection*> projections{&q, &k, &v};

    auto control = [&] {
        for (Projection* p : projections) launch_control(*p, maps, pos, pfslot, m);
    };
    auto candidate = [&] {
        for (Projection* p : projections) launch_candidate(*p, maps, pos, pfslot, m);
    };
    control();
    candidate();
    CK(cudaDeviceSynchronize());

    size_t mismatches = 0;
    double error2 = 0.0, reference2 = 0.0;
    for (Projection* p : projections) {
        const size_t elements = p->out_stride
                                    ? (size_t)2 * p->nhead * slot_rows * HD
                                    : (size_t)m * p->n;
        std::vector<bf16> expected(elements), actual(elements);
        CK(cudaMemcpy(expected.data(), p->control, elements * sizeof(bf16),
                      cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(actual.data(), p->candidate, elements * sizeof(bf16),
                      cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < elements; ++i) {
            mismatches += std::memcmp(&expected[i], &actual[i], sizeof(bf16)) != 0;
            const double ref = __bfloat162float(expected[i]);
            const double err = __bfloat162float(actual[i]) - ref;
            reference2 += ref * ref;
            error2 += err * err;
        }
    }
    const double rel_l2 = std::sqrt(error2 / std::max(reference2, 1e-30));

    for (int warmup = 0; warmup < 2; ++warmup) {
        (void)timed(control);
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
        (void)timed(candidate);
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
    std::vector<float> control_ms, candidate_ms;
    for (int repeat = 0; repeat < 11; ++repeat) {
        if (repeat & 1) {
            candidate_ms.push_back(timed(candidate));
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
            control_ms.push_back(timed(control));
        } else {
            control_ms.push_back(timed(control));
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
            candidate_ms.push_back(timed(candidate));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
    std::sort(control_ms.begin(), control_ms.end());
    std::sort(candidate_ms.begin(), candidate_ms.end());

    std::printf("Gemma-4 BF16 sliding QKV+HNR M%u K%u packed-slots=2 seed=%u\n", m, K,
                seed_arg);
    print_resources("control projection", (const void*)control_projection, 384,
                    CONTROL_SMEM);
    print_resources("fused projection+HNR", (const void*)fused_projection_hnr, 384,
                    FUSED_SMEM);
    std::printf("correctness mismatches=%zu rel-L2=%.3e %s\n", mismatches, rel_l2,
                mismatches == 0 ? "PASS" : "FAIL");
    std::printf("control p50=%.3f ms candidate p50=%.3f ms speedup=%.3fx\n",
                control_ms[control_ms.size() / 2], candidate_ms[candidate_ms.size() / 2],
                control_ms[control_ms.size() / 2] /
                    candidate_ms[candidate_ms.size() / 2]);

    for (Projection* p : projections) {
        cudaFree(p->raw);
        cudaFree(p->control);
        cudaFree(p->candidate);
    }
    cudaFree(activation);
    cudaFree(wq);
    cudaFree(wk);
    cudaFree(wv);
    cudaFree(qgamma);
    cudaFree(kgamma);
    cudaFree(pos);
    cudaFree(pfslot);
    cudaFree(cosb);
    cudaFree(sinb);
    cudaFree(maps);
    return mismatches == 0 ? 0 : 1;
}
