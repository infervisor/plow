#pragma once

static __device__ __forceinline__ unsigned fp8_mma_bf16_pair(unsigned short bytes) {
    const __half2_raw raw = __nv_cvt_fp8x2_to_halfraw2(bytes, __NV_E4M3);
    const float2 values = __half22float2(*reinterpret_cast<const __half2*>(&raw));
    const __nv_bfloat162 result = __floats2bfloat162_rn(values.x, values.y);
    return *reinterpret_cast<const unsigned*>(&result);
}

static __device__ __forceinline__ void fp8_mma_accumulate(
    float (&acc)[4], const unsigned (&a)[4], const uint8_t* weight, bool valid) {
    unsigned b0 = 0, b1 = 0;
    if (valid) {
        b0 = fp8_mma_bf16_pair(*reinterpret_cast<const unsigned short*>(weight));
        b1 = fp8_mma_bf16_pair(*reinterpret_cast<const unsigned short*>(weight + 8));
    }
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

template <bool Glu>
static __device__ void d_gemv_fp8_mma(__nv_bfloat16* __restrict__ C,
    const __nv_bfloat16* __restrict__ x, const uint8_t* __restrict__ W,
    const uint8_t* __restrict__ Wu, const float* __restrict__ scale,
    const float* __restrict__ su, unsigned M, unsigned N, unsigned K,
    unsigned act, unsigned slice, unsigned nblk) {
    const unsigned lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const unsigned group = lane >> 2, pair = (lane & 3) * 2;
    const unsigned per = (N + nblk - 1) / nblk;
    const unsigned first = slice * per, end = min(first + per, N);
    // Keep the packet's blocked output ownership, including partial final tiles.
    for (unsigned col = first + warp * 8; col < end; col += 64) {
        for (unsigned row0 = 0; row0 < M; row0 += 16) {
            const unsigned row = row0 + group;
            const bool weight_valid = col + group < end;
            float gate[4] = {}, up[4] = {};
            for (unsigned kb = 0; kb < K; kb += 16) {
                const unsigned k = kb + pair;
                unsigned a[4] = {};
                if (row < M) {
                    a[0] = *reinterpret_cast<const unsigned*>(x + (size_t)row * K + k);
                    a[2] = *reinterpret_cast<const unsigned*>(x + (size_t)row * K + k + 8);
                }
                if (row + 8 < M) {
                    a[1] = *reinterpret_cast<const unsigned*>(x + (size_t)(row + 8) * K + k);
                    a[3] = *reinterpret_cast<const unsigned*>(x + (size_t)(row + 8) * K + k + 8);
                }
                const size_t offset = weight_valid ? (size_t)(col + group) * K + k : 0;
                fp8_mma_accumulate(gate, a, W + offset, weight_valid);
                if constexpr (Glu) fp8_mma_accumulate(up, a, Wu + offset, weight_valid);
            }
#pragma unroll
            for (unsigned high = 0; high < 2; ++high)
#pragma unroll
                for (unsigned low = 0; low < 2; ++low) {
                    const unsigned m = row + high * 8, n = col + pair + low;
                    if (m < M && n < end) {
                        const float value = gate[high * 2 + low] * scale[n];
                        float result = value;
                        if constexpr (Glu) {
                            const float activated = act == PLOW_ACT_SILU_ ? act_silu(value) : act_gelu_tanh(value);
                            result = activated * (up[high * 2 + low] * su[n]);
                        }
                        C[(size_t)m * N + n] = __float2bfloat16(result);
                    }
                }
        }
    }
}
