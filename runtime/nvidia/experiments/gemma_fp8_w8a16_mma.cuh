#pragma once
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace gemma_fp8_probe {

__device__ __forceinline__ unsigned unpack_pair(unsigned short bytes) {
    const __half2_raw raw = __nv_cvt_fp8x2_to_halfraw2(bytes, __NV_E4M3);
    const float2 values = __half22float2(*reinterpret_cast<const __half2*>(&raw));
    const __nv_bfloat162 result = __floats2bfloat162_rn(values.x, values.y);
    return *reinterpret_cast<const unsigned*>(&result);
}

// E4M3 values convert exactly to BF16. Scale stays outside the K reduction.
template <bool Partial>
__device__ __forceinline__ void w8a16_mma_tile(
    __nv_bfloat16* output, float* partial, const __nv_bfloat16* x,
    const unsigned char* weights, const float* scales,
    unsigned M, unsigned N, unsigned K, unsigned splits,
    unsigned tile_x, unsigned tile_y, unsigned tile_z) {
    const unsigned lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const unsigned group = lane >> 2, pair = (lane & 3) * 2;
    const unsigned row = tile_y * 16 + group;
    const unsigned col = tile_x * 64 + warp * 8;
    const unsigned tiles = K / 16;
    const unsigned chunk = (tiles + splits - 1) / splits;
    const unsigned start = tile_z * chunk;
    const unsigned end = min(start + chunk, tiles);
    float acc[4] = {};
    for (unsigned tile = start; tile < end; ++tile) {
        const unsigned k = tile * 16 + pair;
        unsigned a[4] = {};
        if (row < M) {
            a[0] = *reinterpret_cast<const unsigned*>(x + (size_t)row * K + k);
            a[2] = *reinterpret_cast<const unsigned*>(x + (size_t)row * K + k + 8);
        }
        if (row + 8 < M) {
            a[1] = *reinterpret_cast<const unsigned*>(x + (size_t)(row + 8) * K + k);
            a[3] = *reinterpret_cast<const unsigned*>(x + (size_t)(row + 8) * K + k + 8);
        }
        unsigned b[2] = {};
        if (col + group < N) {
            const auto* w = weights + (size_t)(col + group) * K + k;
            b[0] = unpack_pair(*reinterpret_cast<const unsigned short*>(w));
            b[1] = unpack_pair(*reinterpret_cast<const unsigned short*>(w + 8));
        }
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
#pragma unroll
    for (unsigned high = 0; high < 2; ++high)
#pragma unroll
        for (unsigned low = 0; low < 2; ++low) {
            const unsigned m = row + high * 8, n = col + pair + low;
            if (m < M && n < N) {
                const size_t index = (size_t)m * N + n;
                if constexpr (Partial)
                    partial[(size_t)tile_z * M * N + index] = acc[high * 2 + low];
                else
                    output[index] = __float2bfloat16(acc[high * 2 + low] * scales[n]);
            }
        }
}

template <bool Partial>
__global__ __launch_bounds__(256) void w8a16_mma(
    __nv_bfloat16* output, float* partial, const __nv_bfloat16* x,
    const unsigned char* weights, const float* scales,
    unsigned M, unsigned N, unsigned K, unsigned splits) {
    w8a16_mma_tile<Partial>(output, partial, x, weights, scales, M, N, K, splits,
                            blockIdx.x, blockIdx.y, blockIdx.z);
}

__global__ __launch_bounds__(256) void w8a16_mma_sliced(
    __nv_bfloat16* output, const __nv_bfloat16* x, const unsigned char* weights,
    const float* scales, unsigned M, unsigned N, unsigned K, unsigned nblk) {
    const unsigned tiles_n = (N + 63) / 64;
    for (unsigned slice = blockIdx.x; slice < nblk; slice += gridDim.x)
        if (M <= 8) {
            for (unsigned tile = slice; tile < tiles_n; tile += nblk)
                w8a16_mma_tile<false>(output, nullptr, x, weights, scales, M, N, K, 1,
                                      tile, 0, 0);
        } else {
            const unsigned tiles = tiles_n * ((M + 15) / 16);
            for (unsigned tile = slice; tile < tiles; tile += nblk)
                w8a16_mma_tile<false>(output, nullptr, x, weights, scales, M, N, K, 1,
                                      tile % tiles_n, tile / tiles_n, 0);
        }
}

__global__ void reduce(__nv_bfloat16* output, const float* partial,
    const float* scales, unsigned M, unsigned N, unsigned splits) {
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (size_t)M * N) return;
    float value = 0;
    for (unsigned split = 0; split < splits; ++split)
        value += partial[(size_t)split * M * N + index];
    output[index] = __float2bfloat16(value * scales[index % N]);
}

}
