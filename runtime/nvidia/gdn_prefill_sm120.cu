#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "gdn_prefill_sm120.h"

namespace {

struct alignas(16) Chunk {
    __nv_bfloat16 q[16][128];
    __nv_bfloat16 k[16][128];
    __nv_bfloat16 v[16][16];
    float alpha[16];
    float beta[16];
    __nv_bfloat16 o[16][16];
};

__device__ __forceinline__ void copy16(void* dst, const void* src) {
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(addr), "l"(src) : "memory");
}

__device__ __forceinline__ void load_chunk(Chunk& c, const GdnPrefillArgs& a,
                                         int start, int count, int head, int tile) {
    const int tid = threadIdx.x;
    for (int i = tid; i < count * 16; i += 128) {
        const int t = i / 16, d = (i % 16) * 8;
        const size_t src = (size_t(start + t) * 16 + head / 3) * 128 + d;
        copy16(&c.q[t][d], a.q + src);
        copy16(&c.k[t][d], a.k + src);
    }
    if (tid < count * 2) {
        const int t = tid / 2, d = (tid % 2) * 8;
        const size_t src = (size_t(start + t) * 48 + head) * 128 + tile * 16 + d;
        copy16(&c.v[t][d], a.v + src);
    }
    if (tid < count) {
        c.alpha[tid] = a.alpha[size_t(start + tid) * 48 + head];
        c.beta[tid] = a.beta[size_t(start + tid) * 48 + head];
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
}

__device__ __forceinline__ void warp_sum4(float (&x)[4]) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
        for (int r = 0; r < 4; ++r)
            x[r] += __shfl_down_sync(0xffffffff, x[r], offset);
    }
#pragma unroll
    for (int r = 0; r < 4; ++r)
        x[r] = __shfl_sync(0xffffffff, x[r], 0);
}

} // namespace

extern "C" __global__ __launch_bounds__(128) void gdn_prefill_sm120(GdnPrefillArgs a) {
    __shared__ Chunk chunks[2];
    const int tile = blockIdx.x, head = blockIdx.y, seq = blockIdx.z;
    const int tid = threadIdx.x, lane = tid % 32, row = (tid / 32) * 4;
    const size_t state0 = ((size_t(seq) * 48 + head) * 128 + tile * 16 + row) * 128;
    const int start = int(a.cu_seqlens[seq]), end = int(a.cu_seqlens[seq + 1]);
    float state[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
        for (int r = 0; r < 4; ++r)
            state[r][j] = a.initial[state0 + r * 128 + lane + 32 * j];
    }

    if (start < end) {
        load_chunk(chunks[0], a, start, min(16, end - start), head, tile);
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
    }
    for (int base = start, buffer = 0; base < end; base += 16, buffer ^= 1) {
        Chunk& c = chunks[buffer];
        const int count = min(16, end - base);
        if (base + 16 < end)
            load_chunk(chunks[buffer ^ 1], a, base + 16, min(16, end - base - 16), head, tile);

#pragma unroll 1
        for (int t = 0; t < count; ++t) {
            float q[4], k[4], proj[4] = {}, result[4] = {};
            const float decay = c.alpha[t], beta = c.beta[t];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                q[j] = __bfloat162float(c.q[t][lane + 32 * j]) * a.scale;
                k[j] = __bfloat162float(c.k[t][lane + 32 * j]);
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    state[r][j] *= decay;
                    proj[r] += state[r][j] * k[j];
                }
            }
            warp_sum4(proj);
#pragma unroll
            for (int r = 0; r < 4; ++r)
                proj[r] = (__bfloat162float(c.v[t][row + r]) - proj[r]) * beta;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    state[r][j] = fmaf(proj[r], k[j], state[r][j]);
                    result[r] += state[r][j] * q[j];
                }
            }
            warp_sum4(result);
            if (lane == 0) {
#pragma unroll
                for (int r = 0; r < 4; ++r)
                    c.o[t][row + r] = __float2bfloat16_rn(result[r]);
            }
        }
        // Publish the output tile and the prefetched inputs together. The next
        // chunk's boundary also protects this buffer's stores before its reuse.
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        if (tid < count * 2) {
            const int t = tid / 2, d = (tid % 2) * 8;
            const size_t dst = (size_t(base + t) * 48 + head) * 128 + tile * 16 + d;
            *reinterpret_cast<uint4*>(a.o + dst) = *reinterpret_cast<const uint4*>(&c.o[t][d]);
        }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
        for (int r = 0; r < 4; ++r)
            a.state[state0 + r * 128 + lane + 32 * j] = state[r][j];
    }
}
