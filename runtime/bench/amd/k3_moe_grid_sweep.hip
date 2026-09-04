#include "amd_common.h"
#include "op_moe.h"

#ifndef K3_MOE_GRID_XCD
#define K3_MOE_GRID_XCD 1
#endif

__device__ __forceinline__ unsigned k3_moe_slice() {
    if (K3_MOE_GRID_XCD == 1 || gridDim.x % K3_MOE_GRID_XCD != 0) return blockIdx.x;
    return blockIdx.x % K3_MOE_GRID_XCD * (gridDim.x / K3_MOE_GRID_XCD) +
           blockIdx.x / K3_MOE_GRID_XCD;
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4)
void k3_moe_group_glu(bf16* fu, const bf16* x, const unsigned char* table,
                      const unsigned long long* weights,
                      const unsigned long long* scales, unsigned topk,
                      unsigned intermediate, unsigned hidden, unsigned experts) {
    d_moe_group_glu_fp8_blk(fu, x, table, weights, scales, topk, intermediate, hidden,
                            experts, PLOW_MOE_ACT_SITU, k3_moe_slice(), gridDim.x,
                            PLOW_MOE_ENC_MXFP4, 4.0f, 25.0f);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4)
void k3_moe_group_down(float* partial, const bf16* fu, const unsigned char* table,
                       const unsigned long long* weights,
                       const unsigned long long* scales, unsigned topk,
                       unsigned hidden, unsigned intermediate, unsigned experts) {
    d_moe_group_down_fp8_blk(partial, fu, table, weights, scales, topk, hidden,
                             intermediate, experts, k3_moe_slice(), gridDim.x,
                             PLOW_MOE_ENC_MXFP4);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4)
void k3_moe_combine(bf16* out, const float* partial, unsigned topk,
                    unsigned hidden) {
    d_moe_combine(out, nullptr, nullptr, partial, hidden, topk,
                  blockIdx.x, gridDim.x);
}

/* Sound one-grid GLU -> DOWN experiment. Every routed slot is owned by exactly one physical
 * XCD, and the last producer on each XCD publishes that partition once. Its peers wait on an
 * XCD-local ready word and invalidate only their private L1. All eight leaders cross the global
 * threshold before any DOWN work begins.
 * `control` is monotonic so consecutive launches need no host-side reset:
 * [XCD census:8][arrive:8][ready:8][global_arrive:1]. */
extern "C" __global__ __launch_bounds__(PLOW_THREADS, 2)
void k3_moe_group_pair_xcd(float* partial, bf16* fu, const bf16* x,
                           const unsigned char* table,
                           const unsigned long long* weights,
                           const unsigned long long* scales, unsigned topk,
                           unsigned intermediate, unsigned hidden, unsigned experts,
                           unsigned* control) {
    constexpr unsigned domains = 8;
    const unsigned domain =
        __builtin_amdgcn_s_getreg((20 /* HW_REG_XCC_ID */) | ((32 - 1) << 11)) & 0xf;
    if (domain >= domains || gridDim.x % domains || topk % domains) __builtin_trap();

    const unsigned local_blocks = gridDim.x / domains;
    const unsigned local_topk = topk / domains;
    const unsigned local_block =
        __hip_atomic_fetch_add(&control[domain], 1u, __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT) % local_blocks;
    const size_t slot0 = static_cast<size_t>(domain) * local_topk;

    d_moe_group_glu_fp8_blk(fu + slot0 * intermediate, x, table + slot0 * 8,
                            weights, scales, local_topk, intermediate, hidden,
                            experts, PLOW_MOE_ACT_SITU, local_block, local_blocks,
                            PLOW_MOE_ENC_MXFP4, 4.0f, 25.0f);
    __syncthreads();

    if (threadIdx.x == 0) {
        unsigned* const arrive = &control[domains + domain];
        unsigned* const ready = &control[2 * domains + domain];
        unsigned* const global = &control[3 * domains];
        const unsigned was = __hip_atomic_fetch_add(arrive, 1u, __ATOMIC_RELAXED,
                                                     __HIP_MEMORY_SCOPE_AGENT);
        const unsigned local_target = (was / local_blocks + 1u) * local_blocks;
        if (was + 1u == local_target) {
            const unsigned global_was =
                __hip_atomic_fetch_add(global, local_blocks, __ATOMIC_RELEASE,
                                       __HIP_MEMORY_SCOPE_AGENT);
            const unsigned global_target =
                (global_was / gridDim.x + 1u) * gridDim.x;
            while (__hip_atomic_load(global, __ATOMIC_RELAXED,
                                     __HIP_MEMORY_SCOPE_AGENT) < global_target) {
                __builtin_amdgcn_s_sleep(1);
            }
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
            __hip_atomic_store(ready, local_target, __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT);
        } else {
            while (__hip_atomic_load(ready, __ATOMIC_RELAXED,
                                     __HIP_MEMORY_SCOPE_AGENT) < local_target) {
                __builtin_amdgcn_s_sleep(1);
            }
            asm volatile("buffer_inv" ::: "memory");
        }
    }
    __syncthreads();

    d_moe_group_down_fp8_blk(partial + slot0 * hidden,
                             fu + slot0 * intermediate, table + slot0 * 8,
                             weights, scales, local_topk, hidden, intermediate,
                             experts, local_block, local_blocks, PLOW_MOE_ENC_MXFP4);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS, 2)
void k3_moe_xcd_handoff(unsigned* control) {
    constexpr unsigned domains = 8;
    const unsigned domain =
        __builtin_amdgcn_s_getreg((20 /* HW_REG_XCC_ID */) | ((32 - 1) << 11)) & 0xf;
    if (domain >= domains || gridDim.x % domains) __builtin_trap();
    const unsigned local_blocks = gridDim.x / domains;
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned* const arrive = &control[domains + domain];
        unsigned* const ready = &control[2 * domains + domain];
        unsigned* const global = &control[3 * domains];
        const unsigned was = __hip_atomic_fetch_add(arrive, 1u, __ATOMIC_RELAXED,
                                                     __HIP_MEMORY_SCOPE_AGENT);
        const unsigned local_target = (was / local_blocks + 1u) * local_blocks;
        if (was + 1u == local_target) {
            const unsigned global_was =
                __hip_atomic_fetch_add(global, local_blocks, __ATOMIC_RELEASE,
                                       __HIP_MEMORY_SCOPE_AGENT);
            const unsigned global_target =
                (global_was / gridDim.x + 1u) * gridDim.x;
            while (__hip_atomic_load(global, __ATOMIC_RELAXED,
                                     __HIP_MEMORY_SCOPE_AGENT) < global_target)
                __builtin_amdgcn_s_sleep(1);
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
            __hip_atomic_store(ready, local_target, __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT);
        } else {
            while (__hip_atomic_load(ready, __ATOMIC_RELAXED,
                                     __HIP_MEMORY_SCOPE_AGENT) < local_target)
                __builtin_amdgcn_s_sleep(1);
            asm volatile("buffer_inv" ::: "memory");
        }
    }
    __syncthreads();
}

/* Generic phase candidate for an adjacent grouped-DOWN -> deterministic-COMBINE seam.
 * The full 256-WG producer remains resident; after the sound per-XCD handoff, seven
 * workgroups run the unchanged fixed-slot-order combine. */
extern "C" __global__ __launch_bounds__(PLOW_THREADS, 2)
void k3_moe_down_combine_xcd(
    bf16* out, float* partial, const bf16* fu, const unsigned char* table,
    const unsigned long long* weights, const unsigned long long* scales,
    unsigned topk, unsigned hidden, unsigned intermediate, unsigned experts,
    unsigned* control) {
    if (gridDim.x != 256 || blockDim.x != PLOW_THREADS || topk == 0 || hidden == 0)
        __builtin_trap();
    d_moe_group_down_fp8_blk(partial, fu, table, weights, scales, topk, hidden,
                             intermediate, experts, blockIdx.x, gridDim.x,
                             PLOW_MOE_ENC_MXFP4);
    __syncthreads();

    constexpr unsigned domains = 8;
    const unsigned domain =
        __builtin_amdgcn_s_getreg((20 /* HW_REG_XCC_ID */) | ((32 - 1) << 11)) & 0xf;
    if (domain >= domains) __builtin_trap();
    if (threadIdx.x == 0) {
        unsigned* const arrive = &control[domain];
        unsigned* const ready = &control[domains + domain];
        unsigned* const global = &control[2 * domains];
        const unsigned was = __hip_atomic_fetch_add(arrive, 1u, __ATOMIC_RELAXED,
                                                     __HIP_MEMORY_SCOPE_AGENT);
        const unsigned local_target = (was / 32u + 1u) * 32u;
        if (was + 1u == local_target) {
            const unsigned global_was = __hip_atomic_fetch_add(
                global, 32u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
            const unsigned global_target = (global_was / 256u + 1u) * 256u;
            while (__hip_atomic_load(global, __ATOMIC_RELAXED,
                                     __HIP_MEMORY_SCOPE_AGENT) < global_target)
                __builtin_amdgcn_s_sleep(1);
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
            __hip_atomic_store(ready, local_target, __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT);
        } else {
            while (__hip_atomic_load(ready, __ATOMIC_RELAXED,
                                     __HIP_MEMORY_SCOPE_AGENT) < local_target)
                __builtin_amdgcn_s_sleep(1);
            asm volatile("buffer_inv" ::: "memory");
        }
    }
    __syncthreads();
    if (blockIdx.x < 7)
        d_moe_combine(out, nullptr, nullptr, partial, hidden, topk,
                      blockIdx.x, 7);
}

extern "C" __device__ unsigned plow_moe_down_combine_phase_abi_1 = 1;
extern "C" __device__ unsigned plow_moe_down_combine_phase_grid_256 = 256;
extern "C" __device__ unsigned plow_moe_down_combine_phase_domains_8 = 8;

extern "C" __global__ void k3_moe_xcd_map(unsigned* out) {
    if (threadIdx.x == 0)
        out[blockIdx.x] =
            __builtin_amdgcn_s_getreg((20 /* HW_REG_XCC_ID */) | ((32 - 1) << 11)) & 0xf;
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4)
void k3_moe_stream(float* sink, const unsigned char* arena, unsigned long long bytes) {
    const unsigned long long thread =
        static_cast<unsigned long long>(blockIdx.x) * PLOW_THREADS + threadIdx.x;
    const unsigned long long threads =
        static_cast<unsigned long long>(gridDim.x) * PLOW_THREADS;
    const PLOW_GLOB fp4v32* words =
        (const PLOW_GLOB fp4v32*)(const PLOW_GLOB void*)arena;
    float sum = 0.0f;
    for (unsigned long long i = thread; i < bytes / sizeof(fp4v32); i += threads) {
        const fp4v32 value = words[i];
        sum += static_cast<float>(value[0] ^ value[1] ^ value[2] ^ value[3]);
    }
    if (sum == 1234.5f) sink[0] = sum;
}
