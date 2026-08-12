/* Exact K3 latent-MoE combine -> TP all-reduce boundary.
 *
 * The control recreates the two-packet protocol inside one dispatch: all combine
 * workgroups publish a device-local completion counter, then one workgroup signals
 * the rank-wide cross-GPU gate. The candidate lets every combine workgroup publish
 * its own completed slice through the same local aggregation protocol used by
 * PLOW_XR_AGG, then reduces directly. Both keep the production f32 -> bf16 boundary.
 */
#include "../amd/op_collective.h"
#include "../amd/op_moe.h"

static __device__ __forceinline__ uint32_t local_poll(const uint32_t* p) {
    return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

static __device__ __forceinline__ void local_acquire(void) {
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
}

static __device__ __forceinline__ void local_signal(uint32_t* p) {
    __hip_atomic_fetch_add(p, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
}

static __device__ __forceinline__ bool wait_local(uint32_t* p, uint32_t target,
                                                   uint64_t deadline, uint32_t* status,
                                                   uint32_t code) {
    const uint64_t t0 = __builtin_amdgcn_s_memrealtime();
    while (local_poll(p) < target) {
        if (__builtin_amdgcn_s_memrealtime() - t0 > deadline) {
            if (status) *status = code;
            return false;
        }
        __builtin_amdgcn_s_sleep(2);
    }
    local_acquire();
    return true;
}

static __device__ __forceinline__ bool wait_remote(uint32_t* p, uint32_t target,
                                                    uint64_t deadline, uint32_t* status,
                                                    uint32_t code) {
    const uint64_t t0 = __builtin_amdgcn_s_memrealtime();
    while (xctr_poll(p) < target) {
        if (__builtin_amdgcn_s_memrealtime() - t0 > deadline) {
            if (status) *status = code;
            return false;
        }
        __builtin_amdgcn_s_sleep(2);
    }
    xctr_acquire();
    return true;
}

static __device__ __forceinline__ void finish_rank(uint32_t* done, uint32_t nblk,
                                                    uint64_t deadline, uint32_t* status,
                                                    uint64_t t0, uint64_t* cycles,
                                                    int* bailed) {
    if (threadIdx.x == 0) local_signal(done);
    __syncthreads();
    if (threadIdx.x == 0)
        *bailed = !wait_local(done, nblk, deadline, status, 0xD0E00000u);
    __syncthreads();
    if (*bailed) return;
    if (blockIdx.x == 0 && threadIdx.x == 0 && cycles)
        *cycles = __builtin_amdgcn_s_memrealtime() - t0;
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS)
void tp_moe_combine_xreduce_control(
    bf16* out, const void* const* peer_scratch, uint32_t* const* peer_gate,
    uint32_t* local_gate, const float* part, uint32_t nranks, uint32_t rank,
    uint32_t hidden, uint32_t topk, uint32_t slot_stride, uint32_t iters,
    uint64_t deadline, uint64_t* cycles, uint32_t* status) {
    __shared__ int bailed;
    const uint64_t t0 = __builtin_amdgcn_s_memrealtime();
    const uint32_t nblk = gridDim.x;
    for (uint32_t i = 0; i < iters; i++) {
        const uint32_t slot = i * slot_stride;
        bf16* local = (bf16*)((char*)peer_scratch[rank] + slot);
        d_moe_combine(local, nullptr, nullptr, part, hidden, topk, blockIdx.x, nblk);
        __syncthreads();

        uint32_t* const lc = local_gate + (size_t)i * PLOW_CTR_STRIDE;
        if (threadIdx.x == 0) local_signal(lc);
        __syncthreads();
        if (threadIdx.x == 0)
            bailed = !wait_local(lc, nblk, deadline, status, 0xC0100000u | i);
        __syncthreads();
        if (bailed) return;

        uint32_t* const rg = peer_gate[rank] + (size_t)i * PLOW_CTR_STRIDE;
        if (blockIdx.x == 0 && threadIdx.x == 0)
            for (uint32_t r = 0; r < nranks; r++)
                xctr_signal(peer_gate[r] + (size_t)i * PLOW_CTR_STRIDE);
        if (threadIdx.x == 0)
            bailed = !wait_remote(rg, nranks, deadline, status, 0xC0200000u | i);
        __syncthreads();
        if (bailed) return;
        d_xreduce(out, peer_scratch, slot, nranks, hidden, blockIdx.x, nblk);
    }
    finish_rank(local_gate + (size_t)iters * PLOW_CTR_STRIDE, nblk, deadline, status, t0,
                cycles, &bailed);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS)
void tp_moe_combine_xreduce_fused(
    bf16* out, const void* const* peer_scratch, uint32_t* const* peer_gate,
    uint32_t* local_gate, const float* part, uint32_t nranks, uint32_t rank,
    uint32_t hidden, uint32_t topk, uint32_t slot_stride, uint32_t iters,
    uint64_t deadline, uint64_t* cycles, uint32_t* status) {
    __shared__ int bailed;
    const uint64_t t0 = __builtin_amdgcn_s_memrealtime();
    const uint32_t nblk = gridDim.x;
    for (uint32_t i = 0; i < iters; i++) {
        const uint32_t slot = i * slot_stride;
        bf16* local = (bf16*)((char*)peer_scratch[rank] + slot);
        d_moe_combine(local, nullptr, nullptr, part, hidden, topk, blockIdx.x, nblk);
        __syncthreads();

        uint32_t* const line = peer_gate[rank] + (size_t)i * PLOW_CTR_STRIDE;
        if (threadIdx.x == 0) {
            const uint32_t prev = __hip_atomic_fetch_add(
                line + 1, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
            if (prev + 1u == nblk) {
                xctr_acquire();
                for (uint32_t r = 0; r < nranks; r++)
                    __hip_atomic_fetch_add(
                        peer_gate[r] + (size_t)i * PLOW_CTR_STRIDE, nblk,
                        __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
            }
        }
        __syncthreads();
        if (threadIdx.x == 0)
            bailed = !wait_remote(line, nranks * nblk, deadline, status, 0xF0200000u | i);
        __syncthreads();
        if (bailed) return;
        d_xreduce(out, peer_scratch, slot, nranks, hidden, blockIdx.x, nblk);
    }
    finish_rank(local_gate + (size_t)iters * PLOW_CTR_STRIDE, nblk, deadline, status, t0,
                cycles, &bailed);
}
